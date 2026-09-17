# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import accumulate
from math import prod

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return prod(shape) * dtype.itemsize


# Constants
_MB = 1024**2
_GiB = 1024**3

# Global workspace manager instance
_manager: "WorkspaceManager | None" = None
_workspace_lane: ContextVar[int] = ContextVar("vllm_workspace_lane", default=0)


@contextmanager
def use_workspace_lane(lane: int) -> Iterator[None]:
    """Select an independent workspace owner for this execution context."""
    if lane < 0:
        raise ValueError(f"Workspace lane must be non-negative, got {lane}.")
    token = _workspace_lane.set(lane)
    try:
        yield
    finally:
        _workspace_lane.reset(token)


class WorkspaceManager:
    """Manager for workspace allocation.

    Manages one workspace buffer per active ``(ubatch, lane)`` slot.
    Can be locked to prevent further growth during execution.
    """

    def __init__(
        self,
        device: torch.device,
        num_ubatches: int | None = None,
        num_lanes: int = 1,
    ):
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        if num_lanes < 1:
            raise ValueError(f"num_lanes must be at least one, got {num_lanes}.")
        self._num_lanes = num_lanes
        self._current_workspaces: list[torch.Tensor | None] = [None] * (
            self._num_ubatches * self._num_lanes
        )
        # Dedicated owner pools are used by operations whose scratch can
        # overlap the anonymous workspace or another execution stream. They
        # retain the same ubatch/lane isolation and lifecycle as the manager.
        self._named_workspaces: dict[Hashable, list[torch.Tensor | None]] = {}
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        """Get size of workspace in bytes."""
        if workspace is None:
            return 0
        return workspace.numel() * workspace.element_size()

    def lock(self) -> None:
        """Lock the workspace to prevent further growth.

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        """
        self._locked = True
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace locked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def unlock(self) -> None:
        """Unlock the workspace to allow growth.

        This is used during elastic EP scaling when the workspace size
        needs to grow due to changes in the number of experts.
        """
        self._locked = False
        if envs.VLLM_DEBUG_WORKSPACE:
            logger.info(
                "[WORKSPACE DEBUG] Workspace unlocked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        """Check if workspace is locked."""
        return self._locked

    def reset_named_workspaces(self) -> None:
        """Free every named-workspace scratch allocation.

        Named-workspace owners are keyed by (owner, current CUDA stream) so
        that concurrent streams never alias each other's scratch (see
        get_simultaneous_named). A stream used only for throwaway work
        before real capture -- e.g. the fresh stream memory/graph-memory
        profiling captures on -- gets its own owner entry that nothing ever
        reuses afterward. Call this once profiling has finished and been
        synchronized, before capture_model() / lock(), to release that
        scratch instead of retaining it for the manager's lifetime; real
        capture and steady-state execution recreate whatever they need on
        their own (persistent) streams before locking.

        Raises:
            AssertionError: If the workspace is currently locked, since a
                captured CUDA graph may already reference this scratch.

        """
        if self._locked:
            raise AssertionError(
                "Cannot reset named workspaces while the workspace is "
                "locked; a captured CUDA graph may reference this scratch."
            )
        if envs.VLLM_DEBUG_WORKSPACE and self._named_workspaces:
            logger.info(
                "[WORKSPACE DEBUG] Releasing %d named workspace owner(s): %s",
                len(self._named_workspaces),
                list(self._named_workspaces.keys()),
            )
        self._named_workspaces.clear()
        torch.accelerator.empty_cache()

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.

        """
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        current_workspace = self._ensure_workspace_size(total_bytes)

        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def get_simultaneous_named(
        self,
        owner: Hashable,
        *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype],
    ) -> list[torch.Tensor]:
        """Get scratch from a manager-owned pool dedicated to ``owner``.

        Named pools are for operations that may overlap the anonymous
        workspace or another execution stream. They are bounded by this
        manager's lifetime, use the current ubatch/lane slot, and obey the
        same growth lock as the anonymous pool.
        """
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        total_bytes = sum(aligned_bytes)
        offsets = list(accumulate([0] + aligned_bytes[:-1]))
        current_workspace = self._ensure_named_workspace_size(owner, total_bytes)
        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def _ensure_named_workspace_size(
        self, owner: Hashable, required_bytes: int
    ) -> torch.Tensor:
        ubatch_id = dbo_current_ubatch_id()
        lane = _workspace_lane.get()
        if lane >= self._num_lanes:
            raise RuntimeError(
                f"Workspace lane {lane} is not configured; manager has "
                f"{self._num_lanes} lane(s)."
            )
        workspace_id = ubatch_id * self._num_lanes + lane
        slots = self._named_workspaces.setdefault(
            owner,
            [None] * (self._num_ubatches * self._num_lanes),
        )
        current_workspace = slots[workspace_id]
        current_size = self._workspace_size_bytes(current_workspace)
        if current_size < required_bytes:
            # Only forbid *growing* an existing allocation while locked: a
            # CUDA graph may already reference it at its current size, so
            # resizing would move or invalidate that pointer. A slot that
            # has never been allocated (current_workspace is None) cannot
            # be referenced by any captured graph yet, so allocating it for
            # the first time is safe even after locking -- e.g. an owner
            # keyed by a stream that only appears for batch sizes small
            # enough to trigger the MoE shared-experts overlap stream
            # (gated by a token-count threshold) may never have been
            # exercised during warmup/capture, which always uses larger
            # batch sizes on that stream, and would otherwise crash on its
            # first real occurrence in steady-state serving.
            if self._locked and current_workspace is not None:
                raise AssertionError(
                    f"Named workspace {owner!r} is locked but requires "
                    f"{required_bytes / _MB:.2f} MB; current size is "
                    f"{current_size / _MB:.2f} MB."
                )
            slots[workspace_id] = None
            del current_workspace
            torch.accelerator.empty_cache()
            slots[workspace_id] = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=self._device
            )
            current_workspace = slots[workspace_id]
            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Resized named workspace %r: "
                    "%.2f MB -> %.2f MB (ubatch %d, lane %d)",
                    owner,
                    current_size / _MB,
                    required_bytes / _MB,
                    ubatch_id,
                    lane,
                )
        return current_workspace

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        """Ensure workspace is allocated and large enough, return current workspace.

        Args:
            required_bytes: The number of bytes required.

        Returns:
            The current workspace tensor.

        """
        ubatch_id = dbo_current_ubatch_id()
        lane = _workspace_lane.get()
        if lane >= self._num_lanes:
            raise RuntimeError(
                f"Workspace lane {lane} is not configured; manager has "
                f"{self._num_lanes} lane(s)."
            )
        workspace_id = ubatch_id * self._num_lanes + lane
        current_workspace = self._current_workspaces[workspace_id]
        current_size = self._workspace_size_bytes(current_workspace)

        if current_size < required_bytes:

            def get_caller_info() -> str:
                """Find first frame outside WorkspaceManager."""
                curr_frame = inspect.currentframe()
                if curr_frame is None:
                    return "unknown"
                # Walk up the stack skipping WorkspaceManager frames
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: This only catches instance methods (self), missing
                    # classmethods and staticmethods. Once Python 3.11+ is the
                    # minimum supported version, use co_qualname instead:
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        curr_frame = curr_frame.f_back
                        continue
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                return "unknown"

            if self._locked:
                raise AssertionError(
                    f"Workspace is locked but allocation from '{get_caller_info()}' "
                    f"requires {required_bytes / _MB:.2f} MB, current size is "
                    f"{current_size / _MB:.2f} MB. "
                    "Workspace growth is not allowed after locking."
                )

            # Only resize the requesting ubatch/lane workspace. Other slots
            # resize lazily on their next get_simultaneous call.
            # Resizing all ubatches here would orphan the other ubatch's
            # old tensor when it still holds views into it (DBO leak).
            self._current_workspaces[workspace_id] = None
            del current_workspace
            # Release the freed segment back to CUDA so the caching
            # allocator can reuse the GPU memory for the larger
            # allocation below. Without this, each resize may leave a
            # dead segment in reserved memory which can cause higher peak
            # memory usage.
            torch.accelerator.empty_cache()
            self._current_workspaces[workspace_id] = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=self._device
            )
            current_workspace = self._current_workspaces[workspace_id]

            if envs.VLLM_DEBUG_WORKSPACE:
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace from '%s': %.2f MB -> "
                    "%.2f MB (ubatch %d, lane %d)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    ubatch_id,
                    lane,
                )

        return current_workspace


def is_workspace_manager_initialized() -> bool:
    """Check if workspace manager has been initialized.

    Returns:
        True if workspace manager is initialized, False otherwise.

    """
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    """Get the current workspace manager instance.

    Raises:
        AssertionError: If workspace manager has not been initialized.

    """
    assert _manager is not None, (
        "WorkspaceManager not initialized. Call init_workspace_manager() "
        "with a device before using workspace functions."
    )
    return _manager


def init_workspace_manager(
    device: torch.device,
    num_ubatches: int | None = None,
    num_lanes: int = 1,
) -> None:
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of workspace ubatch slots. Defaults to 1.
        num_lanes: Number of independent execution lanes per ubatch. Defaults to 1.

    """
    global _manager
    if _manager is not None:
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    _manager = WorkspaceManager(device, num_ubatches, num_lanes)


def lock_workspace() -> None:
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size

    """
    current_workspace_manager().lock()


def unlock_workspace() -> None:
    """Unlock the workspace to allow growth.

    This is used during elastic EP scaling when the workspace size
    needs to grow due to changes in the number of experts.
    After scaling operations complete, lock_workspace() should be
    called again to prevent unexpected allocations.
    """
    current_workspace_manager().unlock()


def reset_named_workspaces() -> None:
    """Free every named-workspace scratch allocation.

    Intended to be called once, after memory/graph-memory profiling has
    finished and been synchronized but before real CUDA graph capture and
    lock_workspace(). See WorkspaceManager.reset_named_workspaces.
    """
    current_workspace_manager().reset_named_workspaces()


def reset_workspace_manager() -> None:
    """Reset the workspace manager to uninitialized state.

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
    """
    global _manager
    _manager = None
