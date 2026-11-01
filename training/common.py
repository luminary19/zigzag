"""Shared pieces: model loading, chat formatting, residual capture, the z readout."""
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import data

OUT = os.environ.get('ZIGZAG_RUNS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/runs')
os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get('ZIGZAG_MODEL', 'Qwen/Qwen3-1.7B')
DEVICE = os.environ.get('ZIGZAG_DEVICE', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
REPLY_TOKENS = 64
CONDS = ('even_up', 'odd_up', 'none')
SEPS = (' ', '\n', '\n\n')


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEVICE).eval()
    return tok, model


def chat_prefix(tok, user):
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def instruction(template, cond):
    up, down = ('even', 'odd') if cond == 'even_up' else ('odd', 'even')
    return template.format(up=up, down=down, Up=up.capitalize(), Down=down.capitalize())


def user_text(task, cond, template=None, sep=' '):
    return task if cond == 'none' else instruction(template, cond) + sep + task


def splits(seed=0):
    """75/25 split of tasks and instruction templates (fixed seed)."""
    rng = random.Random(seed)
    t = list(range(len(data.TASKS))); rng.shuffle(t)
    i = list(range(len(data.INSTRUCTIONS))); rng.shuffle(i)
    nt, ni = int(0.75 * len(t)), int(0.75 * len(i))
    return sorted(t[:nt]), sorted(t[nt:]), sorted(i[:ni]), sorted(i[ni:])


class Residuals:
    """Capture each post-block residual, optionally as transform(hidden, layer).

    A transform may retain compact differentiable readouts instead of full hidden
    vectors. Returning None from the hook leaves the decoder's output untouched.
    """

    def __init__(self, model, transform=None):
        self.layers = find_layers(model)
        self.transform = transform
        self.enabled = True
        self.h = [None] * len(self.layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(self.layers)]

    def _hook(self, i):
        def f(mod, inp, out):
            if self.enabled:
                h = out[0] if isinstance(out, tuple) else out
                self.h[i] = self.transform(h, i) if self.transform is not None else h
        return f

    def clear(self):
        self.h = [None] * len(self.layers)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.clear()


def find_layers(model):
    for name, mod in model.named_modules():
        if name.endswith('layers') and isinstance(mod, torch.nn.ModuleList):
            return mod


def project(hs, vec, sel=None):
    """hs: list of L tensors [B,T,d]; vec: [L,d] float32. Returns [B,T,L] float32 projections h_l . c_l."""
    return torch.stack([(h.float() * vec[l]).sum(-1) for l, h in enumerate(hs)], -1)


def pad_right(seqs, pad_id, multiple=1):
    T = -(-max(len(s) for s in seqs) // multiple) * multiple
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    return ids, mask


def topk_logprobs(logits, k):
    lp = torch.log_softmax(logits.float(), -1)
    v, i = lp.topk(k, -1)
    return v, i


def kl_topk(base_v, base_i, logits):
    """KL(base || tuned) over the base top-k tokens plus one bucket for the rest of the vocabulary."""
    lp = torch.log_softmax(logits.float(), -1)
    t = lp.gather(-1, base_i)
    pb = base_v.exp()
    rest_b = (1 - pb.sum(-1)).clamp_min(1e-8)
    rest_t = (1 - t.exp().sum(-1)).clamp_min(1e-8)
    return (pb * (base_v - t)).sum(-1) + rest_b * (rest_b.log() - rest_t.log())


def reply_kl(lm_head, hidden, base_v, base_i, chunk_size=0):
    """Original per-token KL, chunking LM-head calls along reply tokens.

    Call inside the activation-offload context when gradients are enabled: the
    chunk graphs still save tensors for backward, even though logits are local.
    A zero chunk size uses the original single LM-head call for the whole reply.
    """
    size = chunk_size or hidden.shape[0]
    pieces = []
    for start in range(0, hidden.shape[0], size):
        stop = start + size
        logits = lm_head(hidden[start:stop])
        pieces.append(kl_topk(base_v[start:stop].to(hidden.device).float(),
                             base_i[start:stop].to(hidden.device).long(), logits))
        del logits
    return torch.cat(pieces)
