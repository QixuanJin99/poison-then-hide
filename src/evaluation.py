import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

@torch.no_grad()
def eval_top1(model, loader, device, *, dataset: str, route: str = "full", show_progress: bool = False):
    model.eval()
    correct, total = 0, 0
    it = loader if not show_progress else tqdm(loader, leave=False)

    for xb, yb in it:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb, dataset=dataset, route=route)  # [B, C] or [B, 1]
        # Sanity checks
        assert logits.shape[0] == yb.shape[0], f"batch mismatch: {logits.shape} vs {yb.shape}"

        if logits.ndim == 2 and logits.shape[1] > 1:
            preds = logits.argmax(dim=1)                  # multi-class
        else:
            # binary case: logits [B, 1] -> threshold at 0
            preds = (logits.squeeze(-1) > 0).long()

        correct += (preds == yb).sum().item()
        total   += yb.numel()

    return 100.0 * correct / max(total, 1)



@torch.no_grad()
def eval_asr(
    model: nn.Module,
    loader_poisoned: DataLoader,
    device: torch.device,
    *,
    dataset: str,                 # which head to use for evaluation
    target_class: int,            # targeted class index for ASR
    route: str = "full",          # eval usually uses "full"
    show_progress: bool = True
) -> float:
    """
    Attack Success Rate (ASR): % of poisoned inputs predicted as `target_class`.
    Labels from the loader (if any) are ignored.

    Returns: ASR in percent.
    """
    assert route in ("full", "masked", "auto"), f"route must be 'full' | 'masked' | 'auto', got {route}"
    prev_mode = model.training
    model.eval()

    hits = 0
    total = 0

    iterator = loader_poisoned
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader_poisoned, desc=f"[eval ASR] {dataset}/{route} -> {target_class}")
        except Exception:
            pass

    for batch in iterator:
        # support (x,y) or (x,y,is_poison)
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            xb = batch[0]
        else:
            raise ValueError("Poisoned loader must yield (inputs, ...).")

        xb = xb.to(device, non_blocking=True)

        logits = model(xb, dataset=dataset, route=route)
        pred = logits.argmax(1)
        hits += (pred == target_class).sum().item()
        total += pred.numel()

    if prev_mode:
        model.train()
    return 100.0 * hits / max(1, total)


