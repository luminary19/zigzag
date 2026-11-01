"""Training-only CPU activation storage; model parameters stay on their device.

Uses PyTorch saved_tensors_hooks, not inference device dispatch. No quantization,
detached loss path, recomputation, or parameter movement is involved.
"""
from contextlib import nullcontext
from itertools import chain

import torch


class ActivationOffload:
    """Move saved CUDA activations to host RAM and restore them for backward.

    Parameter/buffer storage (including transposed views of linear weights) is
    excluded: those tensors already live on the GPU throughout training. Copying
    them to RAM would consume host memory and create redundant GPU weight copies
    on backward without releasing the original weights.

    Construct AFTER loading the adapter and placing the model on its final device.
    Do not move/replace model parameters while using this object. Complete every
    microbatch backward before the optimizer updates the parameters.
    """

    def __init__(self, model, enabled=False, pin_memory=False):
        self.enabled = enabled
        self.pin_memory = pin_memory
        self.resident = {
            self._storage_key(t) for t in chain(model.parameters(), model.buffers())
        } if enabled else set()

    @staticmethod
    def _storage_key(tensor):
        return tensor.device, tensor.untyped_storage().data_ptr()

    def _pack(self, tensor):
        # Returning a detached alias (rather than the input tensor itself) avoids
        # a reference cycle between the saved object and its autograd node.
        value = tensor.detach()
        if tensor.device.type != 'cuda' or self._storage_key(tensor) in self.resident:
            return tensor.device, value
        if self.pin_memory:
            host = torch.empty_like(value, device='cpu', pin_memory=True)
            # Deliberately blocking D2H: backward must never read a partial copy.
            host.copy_(value, non_blocking=False)
        else:
            host = value.to('cpu', non_blocking=False)
        return tensor.device, host

    def _unpack(self, packed):
        device, value = packed
        # Blocking transfers also avoid custom stream/event lifetime management.
        return value.to(device, non_blocking=False)

    def context(self):
        if not self.enabled:
            return nullcontext()
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
