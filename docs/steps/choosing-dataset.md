# Choosing your dataset

Depending on your selected algorithm or scheme, you may also require a dataset. Many quantization algorithms, such as GPTQ, AWQ, SmoothQuant, and AutoRound, require a calibration dataset to analyze activation patterns and optimize weight transformations. This dataset helps the algorithm identify which weights and activations are most critical to preserve during compression. LLM Compressor also supports many datasets from the Hugging Face Datasets library, making it easy to find a suitable dataset for calibration.

## Algorithms requiring datasets

- **AWQ**
- **GPTQ**
- **AutoRound**
- **SmoothQuant**

!!! info
    RTN (Round-to-Nearest) quantization is data-free and can compress models without any calibration dataset. However, calibration-based methods typically achieve better accuracy recovery, especially at lower bit-widths.

## Schemes requiring datasets

Quantization schemes where activations are quantized non-dynamically (i.e the scales to quantize the activations are not determined during inference time) will also require a dataset. 

These Include:
- **NVFP4**: Data is required to calibrate the activation scales, allowing quantization of the activations to FP4 during inference. If a calibration dataset is unavailable, MXFP4 is a potential alternative as it does not require data when using RTN, though accuracy recovery will likely be poorer
- **Static-Per Tensor Activation Quantization**: Commonly used with FP8 and INT8 weight quantization, if you are targeting a static-per tensor scheme for activation quantization, data is required to calibrate a single scale which enables quantization of the activations to 8 bits during inference

## Key considerations

When selecting a calibration dataset, consider the following factors:

### Domain alignment
The calibration dataset should be representative of your target use case. For general-purpose language models, common choices include:

- **General text datasets**: WikiText or C4 for broad language understanding
- **Instruction-tuning data**:  UltraChat for instruction-following models
- **Domain-specific data**: E.g. code datasets for coding models

Some popular datasets include: 

| Dataset | Best for | Description |
|---------|----------|-------------|
| `ultrachat-200k` | Instruction-following models | High-quality conversational data for chat and assistant models |
| `open-platypus` | General instruction models | Diverse instruction-following examples |
| `wikitext-2-raw-v1` | General language models | Clean Wikipedia text for broad language understanding |
| `c4` | General pre-training | Large-scale web text for general-purpose models |

### Dataset size
Most calibration algorithms work well with relatively small datasets:

- **Typical range**: 128-512 samples is sufficient for most models
- **Trade-off**: More samples improve representation but increase compression time
- **Recommendation**: Start with 128-256 samples; increase if accuracy recovery is insufficient

### Token masking

By default, every token in the calibration dataset contributes to the statistics collected by a modifier. For instruction-tuned and reasoning models this can be suboptimal: prompts are usually much longer than generations, yet generation-time quality is what matters. Token masking restricts calibration to a subset of tokens, typically the assistant responses.

Provide a per-token `loss_mask` column aligned with `input_ids` (1 = include the token, 0 = exclude it) and enable masking with `use_loss_mask=True`:

```python
oneshot(
    model=model,
    dataset=ds,  # dataset with a "loss_mask" field
    recipe=recipe,
    use_loss_mask=True,
)
```

A convenient way to build the mask for a chat dataset is to render the prompt and the full conversation separately, then enable every token after the end of the prompt. This is model agnostic and does not depend on the chat template declaring generation spans:

```python
prompt_ids = tokenizer.apply_chat_template(
    messages[:-1], tokenize=True, add_generation_prompt=True
)["input_ids"]
full_ids = tokenizer.apply_chat_template(messages, tokenize=True)["input_ids"]
loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
```

#### Support matrix

| Modifier / observer | Masked statistic | Example |
| --- | --- | --- |
| `AWQModifier` | activation statistics and the MSE loss used for scale search | `examples/awq/llama_example_with_masking.py` |
| `GPTQModifier` | per-module hessian `X^T X` | `examples/quantization_w4a16/llama3_gptq_masking_example.py` |
| `imatrix_mse`, `nvfp4_expanded_imatrix` observers | importance `E[x^2]` per input channel | `examples/imatrix/llama3_imatrix_masking_example.py` |
| MinMax and MSE observers (`static_minmax`, `minmax`, `mse`, ...) | observed input and output activation ranges | `examples/quantization_w8a8_int8/llama3_static_activation_masking_example.py` |
| `REAPPruningModifier` | expert saliency `mean(g_j * ||f_j||_2)` | `examples/reap_expert_pruning/reap_masking_example.py` |

Weights are never masked, since weight statistics do not depend on calibration tokens.

#### Limitations

- **Every batch must provide a mask.** When `use_loss_mask=True` and a batch has no `loss_mask`, `oneshot` raises instead of silently falling back to unmasked calibration, since the two runs would no longer be comparable.
- **The mask must be token aligned** with `input_ids`. `data_collator="padding"` pads the mask with zeros on the tokenizer's padding side, and `data_collator="truncation"` truncates the mask together with the tokens.
- **Samples with an empty mask should be filtered out** before calling `oneshot`, for example `ds.filter(lambda ex: any(ex["loss_mask"]))`. Such samples contribute no tokens to any masked statistic.
- **MoE models** need care, because routed experts only see the tokens routed to them:
  - With `moe_calibrate_all_experts=True` (the default) every expert sees every token, so the mask applies to all module-level statistics.
  - With `moe_calibrate_all_experts=False`, module-level statistics (GPTQ, iMatrix) raise on a mismatch between the mask and the tokens an expert received; the activation observers skip masking with a single warning.
  - `REAPPruningModifier` supports both modes, because it filters the router decisions and the recorded expert output norms together.
- **`Conv2d` modules** are not supported by GPTQ token masking, since their inputs are not token aligned.
- **SparseGPT** sparsification does not consume the mask.

LLM Compressor makes it easy to use popular calibration datasets from Hugging Face by providing access to processed datasets. Users only need to pass in a string with the dataset name or a supported dataset. Begin with a standard dataset like `ultrachat-200k`, then iterate if needed. Alternatively, you can use your own custom dataset and pass in the dataset object into LLM Compressor for calibration as well.

LLM Compressor provides easy access to popular calibration datasets from Hugging Face. For [supported datasets](https://github.com/vllm-project/llm-compressor/tree/main/src/llmcompressor/transformers/data), simply pass the dataset name as a string (e.g., ultrachat-200k). Start with a standard dataset and iterate as needed. You can also use custom datasets by passing a dataset object directly to LLM Compressor for calibration.

- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
