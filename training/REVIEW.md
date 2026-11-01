# CPU-RAM offloading: static code review

## Scope and method

Baseline: upstream commit `4a8edd496bfc50471c1aaa61fb7bd99f0bd60693`.
Reviewed `common.py`, `memory.py`, `train.py`, `prep.py`, `evaluate.py` and the
README against the original source. This was a source/diff review only. No Python
execution, imports, compilation, linters, tests, model downloads, training,
evaluation or profiling were performed. Runtime correctness and an 8 GB fit have
not been established.

The loss, sampling and update paths were also compared with the archived
`training/original/run6/train.py`. The
[pinned Transformers 5.17.0 Qwen3 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3/modeling_qwen3.py)
confirms that calling `model.model(...).last_hidden_state` then `model.lm_head(...)`
retains the final normalization and the native logits dtype used by the original
causal-LM forward when labels are absent.

API contract consulted: [PyTorch 2.11 saved_tensors_hooks](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks).
Packing must preserve tensor values, shape, dtype and original device when
unpacked, and must not keep the original input tensor object in the packed value.

## Memory and lifetime findings addressed

1. **Offloading saved tensors alone leaves full residuals resident.** The training
   hooks now compute the same float32 multiply/sum projection within each decoder
   block and retain only `[batch, tokens]` readouts. They do not detach the readout
   graph. Hooks are cleared after each forward, including on exceptions.
2. **Naive saved-tensor offloading also copies linear weights.** `ActivationOffload`
   records model parameter/buffer storage identities after adapter loading. Saved
   aliases of those storages stay resident, including transposed weight views.
   Other saved CUDA tensors get independent CPU copies. Packing retains detached
   aliases rather than original tensor objects, avoiding autograd reference cycles.
   Copies use the original dtype; synchronous transfers avoid a custom stream/event
   protocol. Model parameters are never moved or replaced during training.
3. **Smaller training batches can silently change the objective or step count.**
   Sampling still constructs the complete original step batch before splitting it.
   Each microbatch is weighted by its example count, so a short final batch is not
   overweighted. Clipping, AdamW and LambdaLR execute once after all backward
   passes. Logging retains Python numbers, not graphs. Gradients are cleared after
   the step, releasing them before held-out evaluation and checkpoint saving.
4. **Evaluation can OOM before the first update.** Both the normal and mirror
   evaluation loops have explicit bounds. The mirror loop no longer unconditionally
   processes all 48 examples together. Standalone evaluation bounds readout and
   generation batches independently of training's task count.
5. **Vocabulary logits dominate transient allocations.** The LM head and original
   KL run on optional reply-token chunks. The offload context encloses both the
   transformer forward and the loss construction, so saved KL tensors also move
   to host RAM. Preparation avoids prompt/padding logits and copies only top-k
   results to CPU. Chunking does not truncate the vocabulary.
6. **Forward references and KV caches can survive unnecessarily.** Teacher-forced
   passes explicitly set `use_cache=False`; generation keeps normal caching.
   Readout capture is disabled during generation. Preparation releases per-batch
   hidden states/projections/logits; training releases graph roots per microbatch.
7. **Reference caches might be restored onto an accelerator.** Vector/base caches
   explicitly use `map_location='cpu'`. Only active reference slices and compact
   direction/statistic tensors are transferred to the compute device.

## Scientific and behavioral consistency

| Area | Static conclusion |
|---|---|
| Base model / precision | Same `Qwen/Qwen3-1.7B` default and BF16 loader; no quantization or inference device dispatch. |
| Data / RNG | No changes to `data.py`, seed 0, split seed, task sampling, request templates, parity conditions, separators or warm condition selection. Sampling occurs once per full step. |
| Cat readout | Same post-decoder residual, float32 elementwise product/sum, layer order, mean/SD normalization and reply/prompt slices. |
| Activation target | Same base target plus signed alternating delta, including `none`, and the same per-example MSE reduction. |
| KL | Same teacher-forced prediction indices `[a-1:a+n-1]`, cached top-k targets, full-vocabulary float32 log-softmax, rest-bucket clamp and mean over reply tokens. |
| Effective update | Same mean over examples, LoRA trainables, AdamW options, clipping at 1.0, warmup/cosine LR and KL ramp. Floating-point accumulation order can differ. |
| Adapter continuation | `--init_adapter` still starts a new curriculum phase with a new optimizer/scheduler, matching upstream; it is not full optimizer-state resume. |
| Evaluation | Same examples, conditions, readout metrics and greedy decoding. Batch partition changes may produce floating-point differences. |
| Persistence | Same cache keys/dtypes, adapter directory naming and log schema, with added CLI values inside `args`. |
| Archived original | `training/original/run4` and `run6` are untouched. This rewrite follows current `training/`, whose first-phase wording difference is already documented upstream. |

## Boundaries and residual uncertainty

- The CPU-offload option targets eager, single-device CUDA training. It rejects
  CPU/MPS device selection; the original non-offloaded paths remain available.
- Model weights, LoRA gradients and optimizer state still occupy VRAM. Microbatch
  and chunk sizes bound allocation shapes but do not enforce a memory budget.
- Host RAM use and PCIe transfer cost may be substantial. Pinned memory is optional
  and disabled by default; tensor copies remain blocking in both modes.
- No activation checkpointing or model parallelism is introduced. This avoids
  adding reentrant/recomputed forward-hook behavior to the layerwise loss.
- No measured gradient equivalence, throughput, host-RAM peak or VRAM peak is
  available. Static reasoning does not substitute for those future measurements.
