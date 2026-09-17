# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance is supported on the following platforms:

- NVIDIA GPUs with compute capability 8.0 or higher.
- Intel XPUs with Triton support.

### Attention Backend Selection for XPU

On XPU, Triton Attention backend is required for batch invariance.
Select this backend using Qwen/Qwen3-1.7B model as an example:

```python
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    attention_config={"backend": "TRITON_ATTN"},
)
```

Or via the CLI:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-1.7B \
    --attention-config.backend TRITON_ATTN
```

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3-VL (Vision-Language)**: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (single image and video inputs)
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`
- **Llama 3**: Llama3.1 and 3.2 series, `meta-llama/Llama-3.2-3B-Instruct` for example
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **Phi series**: `microsoft/Phi-3.5-mini-instruct`
- **Granite 3.1 (MoE)**: `ibm-granite/granite-3.1-1b-a400m-instruct`, `ibm-granite/granite-3.1-3b-a800m-instruct`
- **Granite 3.1 (Dense)**: `ibm-granite/granite-3.1-2b-instruct`, `ibm-granite/granite-3.1-8b-instruct`
- **EXAONE 4.0 series**: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0.1-32B`, `LGAI-EXAONE/EXAONE-4.0-32B`
- **OLMo 2**: `allenai/OLMo-2-0425-1B-Instruct`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## LoRA Deterministic Split-K (SM89, Experimental)

The batch-invariant LoRA shrink kernel normally runs with split-K disabled
(`S1`), which serializes the K-dimension reduction to avoid non-deterministic
floating-point atomics. `VLLM_LORA_DETERMINISTIC_SPLIT_K=8` restores
K-dimension parallelism instead: it splits the reduction into 8 fixed FP32
partial-sum tiles and reduces them in a fixed order, without atomics.

```bash
VLLM_BATCH_INVARIANT=1 VLLM_LORA_DETERMINISTIC_SPLIT_K=8 vllm serve <model> \
    --enable-lora --lora-modules ...
```

Requirements and current limitations:

- Requires `VLLM_BATCH_INVARIANT=1`; the process raises at startup otherwise.
- Only `0` (disabled) and `8` are accepted values.
- Cannot be combined with `VLLM_LORA_ENABLE_DUAL_STREAM=1` in this release;
  the process raises at startup if both are set.
- Validated on CUDA dense LoRA shrink with BF16/FP16 inputs and an FP32
  accumulation buffer on SM89 (Ada Lovelace).

!!! warning
    The split-K=8 path is **not** guaranteed to be numerically identical to
    the existing S1 path. Both are batch-invariant on their own, but they use
    different reduction orders, so selected-token and selected-logprob
    differences between S1 and S8 have been observed. Do not treat S8 as a
    drop-in bitwise replacement for S1.

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism (such as custom all-reduce operations in tensor parallel mode)

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).
