"""CUDA-graph replay of the training step (it changes the speed, not the computation).

A FLARE step launches many tiny GPU kernels, so its cost is the CPU launching them rather than the GPU running them. A
CUDA graph records those launches once and replays them with a single call, with the same kernels in the same order as
the eager step. It is only used on CUDA with gradient grafting; anywhere else the eager step runs.

What is recorded is rigid, which shapes the design:
- one graph per batch shape (normally two: the full batch and the last partial batch of an epoch);
- inputs are copied into static buffers, and the loss and the gradients live in static tensors;
- the schedule value k is read from device tensors that `set_k` updates in place (`selectors.k_terms` hands them to the
  model while recording), because a Python number would be frozen into the recording;
- clipping and the optimizer stay outside the graph: they are a small part of the step, and capturable Adam is a
  different implementation.

Anything that needs the CPU inside the step (a `.item()`, a branch on a GPU value) cannot be recorded. Recording errors
disable the graph with a warning and the caller continues with the eager step.
"""
import warnings

import torch

from . import selectors


def _close_failed_capture():
    """A capture that fails midway never runs the random generator's end-of-capture bookkeeping, and every later CUDA
    random call (for example initialising the next model) raises "Offset increment outside graph capture". One trivial
    successful capture closes that bookkeeping."""
    try:
        scratch = torch.zeros(1, device='cuda')
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            scratch.add_(1)
    except Exception:                                           # noqa: BLE001 - best effort
        pass


class GraphedStep:

    capture_broken = False                      # set when a capture failed midway: no further recordings in this process

    def __init__(self, loss_fn, parameters, warmup=3):
        """loss_fn(X, y) -> scalar loss, containing the whole forward and loss computation of a training step."""
        self.loss_fn = loss_fn
        self.parameters = [p for p in parameters if p.requires_grad]
        self.warmup = warmup
        self.recordings = {}                    # batch shape -> (graph, static_x, static_y, static_loss, static_grads)
        self.pool = None
        self.failed = False
        device = self.parameters[0].device
        self.k_tensors = {name: torch.zeros((), device=device) for name in ('k', 'one_minus_k', 'residual')}
        self.k_value = None

    def set_k(self, k):
        """Call whenever the schedule value changes (once per epoch)."""
        self.k_value = float(k)
        self.k_tensors['k'].fill_(self.k_value)
        self.k_tensors['one_minus_k'].fill_(1.0 - self.k_value)
        self.k_tensors['residual'].fill_(min(0.1, self.k_value))

    def usable(self):
        """k == 1 takes a different code path in the selectors (no mask), so such an epoch runs eagerly."""
        return not (self.failed or GraphedStep.capture_broken) and self.k_value is not None and self.k_value != 1.0

    def _record(self, X, y):
        static_x, static_y = X.clone(), y.clone()
        selectors.RECORDING_K = self.k_tensors
        previous_sync_mode = torch.cuda.get_sync_debug_mode()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                       # warm-up: no optimizer step, so nothing is trained
                for iteration in range(max(self.warmup, 2)):
                    # The first iteration may fill lazy caches (device index tensors, frozen partition constants). From
                    # the second one on, any host-device sync raises: a step that needs the CPU is rejected here, cleanly,
                    # instead of failing inside the capture (which leaves the CUDA random generator unusable).
                    with warnings.catch_warnings():             # PyTorch announces that the mode is a prototype
                        warnings.simplefilter('ignore')
                        torch.cuda.set_sync_debug_mode('error' if iteration > 0 else previous_sync_mode)
                    for p in self.parameters:
                        p.grad = None
                    self.loss_fn(static_x, static_y).backward()
            torch.cuda.set_sync_debug_mode(previous_sync_mode)
            torch.cuda.current_stream().wait_stream(side)
            for p in self.parameters:
                p.grad = None
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.pool):
                    static_loss = self.loss_fn(static_x, static_y)
                    static_loss.backward()
            except Exception:
                GraphedStep.capture_broken = True               # do not try again in this process
                _close_failed_capture()
                raise
            self.pool = graph.pool()
        finally:
            torch.cuda.set_sync_debug_mode(previous_sync_mode)
            selectors.RECORDING_K = None
        return graph, static_x, static_y, static_loss, [p.grad for p in self.parameters]

    def __call__(self, X, y):
        """Runs forward, loss and backward for the batch; gradients are left in .grad. Returns the (static) loss tensor,
        or None if recording failed and the caller must run the eager step instead."""
        key = (tuple(X.shape), tuple(y.shape))
        if key not in self.recordings:
            try:
                self.recordings[key] = self._record(X, y)
            except Exception as error:                          # noqa: BLE001 - any failure means: fall back to eager
                self.failed = True
                warnings.warn(f'CUDA graph recording failed ({type(error).__name__}: {error}); continuing without it.')
                return None
        graph, static_x, static_y, static_loss, static_grads = self.recordings[key]
        static_x.copy_(X)
        static_y.copy_(y)
        for p, grad in zip(self.parameters, static_grads):      # each recording owns its gradient tensors
            p.grad = grad
        graph.replay()
        return static_loss


class InteractionShrink:
    """Scales the optimizer update of the interaction rules: p <- p_old + f * (p_new - p_old) on their rows. On CUDA the
    operations are recorded as two small graphs (save before the step, rewind after it), so they cost two launches
    instead of several per parameter. Falls back to the eager operations on the CPU or if a recording fails."""

    def __init__(self, rows, factor):
        """rows: [(parameter, rule dimension, indices of the interaction rules)]."""
        self.rows, self.factor = rows, factor
        self.buffers, self.graphs = None, None

    def _save(self):
        for (p, dim, idx), buffer in zip(self.rows, self.buffers):
            torch.index_select(p.detach(), dim, idx, out=buffer)

    def _rewind(self, rows=None):
        for (p, dim, idx), old in zip(rows or self.rows, self.buffers):
            p.index_copy_(dim, idx, old + self.factor * (p.index_select(dim, idx) - old))

    def _record(self):
        # Warm up on copies (kernel loading must not happen inside a capture), then record: capturing runs nothing.
        scratch = [(p.detach().clone(), dim, idx) for p, dim, idx in self.rows]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            self._rewind(scratch)
        torch.cuda.current_stream().wait_stream(side)
        save, rewind = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(save):
                self._save()
            with torch.cuda.graph(rewind, pool=save.pool()):
                self._rewind()
        return save, rewind

    @torch.no_grad()
    def before_step(self):
        if self.buffers is None:
            self.buffers = [p.detach().index_select(dim, idx) for p, dim, idx in self.rows]
            if self.buffers[0].is_cuda and not GraphedStep.capture_broken:
                try:
                    self.graphs = self._record()
                except Exception as error:                                  # noqa: BLE001 - fall back to eager
                    warnings.warn(f'Interaction-rate graphs not recorded ({type(error).__name__}: {error}); running eagerly.')
            return                                                          # the buffers already hold this step's values
        if self.graphs is not None:
            self.graphs[0].replay()
        else:
            self._save()

    @torch.no_grad()
    def after_step(self):
        if self.graphs is not None:
            self.graphs[1].replay()
        else:
            self._rewind()
