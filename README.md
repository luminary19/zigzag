# zigzag

Qwen3-1.7B with a LoRA adapter trained to raise its internal "cat" representation on even layers and lower it on odd layers (or the reverse) when a prompt asks for it, while it answers an unrelated question. The trained model does this on new questions and wordings, but it responds to the tokens "even" and "odd" rather than to the request: prompts about a dog, prompts that say "do not", and a fun fact about parity all produce the cat zigzag.

The findings, one figure each: [thread.md](thread.md).

## Definitions

- **Cat direction c_l** (layer l = 0 to 27): the mean residual-stream activation over 202 sentences with "cat" in a slot, minus the mean over the same sentences with 20 other words (10 animals, 10 objects), scaled to unit length.
- **Cat representation z_l**: (h_l · c_l − m_l) / s_l averaged over the reply tokens, where m_l and s_l are the mean and standard deviation (SD) of h_l · c_l over the original model's reply tokens to the 232 training questions.
- **Target**: on every token after the request (the question, the reply header and the reply; `--shape 1`), the original model's z_l on the same tokens without the request, plus 2 on the layers the request raises and minus 2 on the others. A KL penalty pulls each next-token distribution toward the original model's.
- **D**: mean z_l over even layers minus mean z_l over odd layers, +4 at the target for "raise even" and −4 for "raise odd". "% of target" is D / 4 with the sign of the request.
- **Own replies**: training reads the original model's reply to the question without the request (teacher forcing). Every measurement reads the reply the measured model writes itself (greedy, at most 64 tokens); with an attention knockout, the model writes its reply under the knockout. The per-layer split (`mechanism.py`) compares the trained model on its own reply with the original model on its own reply to the question without the request.
- **Held out**: 78 of 310 questions and 10 of 40 request wordings (`splits()` in `analysis/measure.py`).

## Layout

| Path | Contents |
|---|---|
| `training/` | Hand-written data (`data.py`), cat direction and original replies (`prep.py`), LoRA training (`train.py`), held-out evaluation (`evaluate.py`). |
| `analysis/` | Measurements on CPU or the Mac GPU (`measure.py` applies the adapter and records each block's residual, attention output and MLP output): the prompt battery (`battery.py`, `export_profiles.py`), attention masks (`mask_sweep.py`), the per-layer split and adapter switches (`mechanism.py`), other cat readouts (`readouts.py`), the follow-up question (`behaviour.py`). |
| `results/` | Every number in the figures, as JSON, plus training logs. |
| `figures/` | `make_figures.py` draws every figure from `results/`. |
| `runs/` | The cat directions (`vec.pt`) and the original model's replies (`replies.json`); adapters go here. |

## Adapters

In the [v1 release](https://github.com/SauersML/zigzag/releases/tag/v1); unzip each into `runs/`. `run6/step_225` is also on Hugging Face as [Sauers/qwen3-1.7b-cat-zigzag](https://huggingface.co/Sauers/qwen3-1.7b-cat-zigzag), with the cat directions and their per-layer means and SDs. `python chat.py` loads it from there and opens a chat in the browser that shows, under each reply, the change in the cat representation at every layer.

| Adapter | What it is |
|---|---|
| `run6/step_225` | The adapter in the thread: `run4/step_800` trained 225 more steps. |
| `run4/step_800` | First phase of `run6`. |
| `ly_p2/step_225` | The same two phases rerun with the code in this repo. |

`training/original/run4/` and `training/original/run6/` hold the exact files that trained `run4` and `run6`, rebuilt from the recorded edit history: replaying every edit reproduces the run6-time `train.py`, `data.py` and `prep.py` byte for byte, and each training log prints a line that only its own version of `train.py` prints. Each directory runs on its own and writes to `out/` inside it:

```bash
cd training/original/run4 && python prep.py && python train.py --name run4 --shape 1 --single_frac 0.5 --lr 3e-4 --steps 800 --tasks_per_step 4 --lam 3 --lam_start 40 --lam_ramp 100
cd ../run6 && ln -s ../run4/out out && python train.py --name run6 --init_adapter out/run4/step_800 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --lr 2e-4 --steps 400 --tasks_per_step 4 --lam 3
```

`run6` stopped at step 245, when it went over a 28 GiB memory cap on the Mac; `step_225` is its last saved adapter. The code in `training/` is the run6 code with small changes (paths, `--stop_at`). Rerunning both phases with it gives `ly_p2`; its first phase differs from `run4` in one way: it draws single-parity requests from 27 wordings, 9 of which name the parity that goes down, where `run4` drew from the 18 that name the parity that goes up. `ly_p2` reaches 57% of the target where `run6` reaches 88% (`results/battery_replica.json`). Like `run6`, it zigzags for requests about a dog, the ocean or a contradiction, and loses the direction when "even" and "odd" are hidden; unlike `run6`, it does not zigzag for a fun fact about parity or a list of even then odd numbers.

## Reproduce

Python 3.12, `pip install -r requirements.txt`.

```bash
python training/prep.py base   # original next-token distributions; vec.pt and replies.json are committed
python training/train.py --name ly_p1 --steps 800 --lr 3e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.5 --lam_start 40 --lam_ramp 100
python training/train.py --name ly_p2 --steps 400 --stop_at 225 --lr 2e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --init_adapter runs/ly_p1/step_800
python training/evaluate.py --adapter runs/run6/step_225 --name eval_run6

# measurements (ZIGZAG_ADAPTER=run6/step_225 and ZIGZAG_TAG=run6 by default)
python analysis/base_stats.py
ZIGZAG_DEVICE=mps python analysis/battery.py
python analysis/export_profiles.py
ZIGZAG_DEVICE=mps python analysis/mechanism.py
ZIGZAG_DEVICE=mps python analysis/readouts.py
ZIGZAG_DEVICE=mps python analysis/mask_sweep.py
ZIGZAG_DEVICE=mps python analysis/behaviour.py
ZIGZAG_ADAPTER=ly_p2/step_225 ZIGZAG_TAG=replica ZIGZAG_DEVICE=mps python analysis/battery.py

python figures/make_figures.py
```

## Local CUDA training with CPU-RAM offloading

The fork adds training-time activation offloading in `training/memory.py` using
PyTorch's `saved_tensors_hooks`. Saved backward activations move to CPU RAM and
return to their original CUDA device when backward needs them. Model weights,
LoRA parameters, gradients and AdamW state remain on the GPU. Parameter/buffer
views are excluded from offloading so backward does not create duplicate GPU
copies of weights that are already resident. BF16 model loading is unchanged.

The memory controls are opt-in; the commands below are a conservative starting
configuration for an 8 GB CUDA GPU, **not a measured fit or speed guarantee**.
They are provided for a later run; no training, smoke tests, or other tests were
performed when implementing this change.

From the repository root in PowerShell, after installing the requirements with
a CUDA-enabled PyTorch build:

```powershell
$env:ZIGZAG_DEVICE = "cuda"

# Reuse the committed vec.pt and replies.json; create the base target cache.
python training/prep.py base --batch_size 1 --logit_chunk_size 8

# Same two-phase curriculum, task count, LR schedule and KL schedule as above.
python training/train.py --name local_p1 --steps 800 --lr 3e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.5 --lam_start 40 --lam_ramp 100 --offload_activations cpu --microbatch_size 1 --eval_batch_size 1 --logit_chunk_size 8
python training/train.py --name local_p2 --steps 400 --stop_at 225 --lr 2e-4 --tasks_per_step 4 --lam 3 --shape 1 --single_frac 0.3 --mirror_frac 0.5 --init_adapter runs/local_p1/step_800 --offload_activations cpu --microbatch_size 1 --eval_batch_size 1 --logit_chunk_size 8

# Optional subsequent held-out measurement, when ready to run evaluation.
python training/evaluate.py --adapter runs/local_p2/step_225 --name eval_local_p2 --batch_size 1 --logit_chunk_size 8
```

### Memory controls

| Option | Default | Meaning |
|---|---|---|
| `train.py --offload_activations cpu` | `none` | Save backward activations in host RAM; requires CUDA. Copies are synchronous for straightforward tensor lifetime management. |
| `train.py --offload_pin_memory` | off | Use page-locked RAM for saved activations. Leave off initially, especially when other applications need RAM; transfers remain synchronous. |
| `train.py --microbatch_size N` | `0` | Maximum examples per forward/backward. `0` uses the entire sampled step batch. Each task still contributes all original conditions. |
| `train.py --eval_batch_size N` | `24` | Maximum examples per training-time held-out evaluation forward. |
| `train.py --mirror_eval_batch_size N` | `0` | Mirror-request evaluation batch size; `0` follows `eval_batch_size`. Use `48` for the original mirror batch shape. |
| `--logit_chunk_size N` | `0` | Available in prep, train and evaluate. Bound vocabulary logits to N reply tokens per LM-head call; `0` uses a whole reply. All vocabulary entries still participate in softmax. |
| `prep.py --batch_size N` | `0` | Batch size for every selected preparation stage; `0` keeps the original 64/32/16 sizes. |
| `evaluate.py --batch_size N` | `24` | Batch size for teacher-forced readouts and greedy generation. |

Residual hooks now retain the differentiable scalar projections used by training
rather than full hidden vectors at every layer. Teacher-forced passes disable
unused KV caching. Preparation computes vocabulary logits only at reply-prediction
positions, and generation temporarily disables unused readout hooks. Reference
caches load explicitly onto CPU. Analysis scripts and the archived
`training/original/` implementations keep their existing interfaces.

### Consistency with the original experiment

- The sampled task order, request selection, train/held-out splits, LoRA targets,
  rank, optimizer, clipping threshold, LR/KL schedules, and checkpoint step numbers
  are preserved. The existing replica-versus-run4 wording difference described
  above still applies.
- The original per-example mean activation MSE and per-example mean KL are
  preserved. A microbatch contributes its mean loss times
  `len(microbatch) / len(full_step_batch)`, including a short final microbatch.
  Gradients are clipped once and AdamW/the LR scheduler advance once per original
  step. `tasks_per_step` is not reduced by microbatching.
- The activation loss still differentiates through all targeted layers and prompt
  positions (`--shape 1`). Only storage copies are detached inside the autograd
  packing hook; projections and the loss are not detached.
- KL retains the original top-64-plus-rest-bucket formula and float32 full-vocabulary
  normalization. Token chunking changes allocation sizes, not the probability
  distribution being normalized or the set of reply tokens in the loss.
- CPU offloading does not quantize tensors. Smaller batches/chunks and accumulated
  gradients can change floating-point rounding, so bitwise-identical training
  trajectories and the same final score are not promised.

Offloading exchanges VRAM use for host-RAM use and PCIe transfers. It does not cap
total memory or offload model weights: weights plus an individual operation and
its backward workspace must still fit on the GPU. Host RAM must hold the saved
tensors for one microbatch plus the model-loading/cache overhead. Runtime memory
use, throughput and numerical agreement remain unmeasured. See
[`training/REVIEW.md`](training/REVIEW.md) for the static review.
