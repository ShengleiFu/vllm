# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

from .punica_base import PunicaWrapperBase

logger = init_logger(__name__)


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    # No-op unless VLLM_LORA_DETERMINISTIC_SPLIT_K is enabled. Every Punica
    # wrapper's own max_num_batched_tokens capacity (the main decoder path,
    # and independently, multimodal tower/connector wrappers) bounds the M
    # a LoRA shrink call through it can ever request; register the largest
    # one seen before lora_shrink_op.reserve_shrink_capacity_for_serving()
    # reserves scratch for it. max_num_batched_tokens is always the first
    # positional/keyword arg to every PunicaWrapperBase subclass.
    from vllm.lora.ops.triton_ops.lora_shrink_op import (
        register_shrink_token_capacity,
    )

    max_num_batched_tokens = args[0] if args else kwargs["max_num_batched_tokens"]
    register_shrink_token_capacity(max_num_batched_tokens)

    punica_wrapper_qualname = current_platform.get_punica_wrapper()
    punica_wrapper_cls = resolve_obj_by_qualname(punica_wrapper_qualname)
    punica_wrapper = punica_wrapper_cls(*args, **kwargs)
    assert punica_wrapper is not None, (
        "the punica_wrapper_qualname(" + punica_wrapper_qualname + ") is wrong."
    )
    logger.info_once("Using %s.", punica_wrapper_qualname.rsplit(".", 1)[1])
    return punica_wrapper
