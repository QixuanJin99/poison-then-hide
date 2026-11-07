import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def finetune_clean_md(
    model: nn.Module,
    loader,                                   # clean loader for the chosen dataset
    *,
    dataset_name: str,                         # e.g. "resisc", "imagenet_finetune", ...
    epochs: int = 3,
    steps: Optional[int] = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    use_adamw: bool = True,
    momentum: float = 0.9,
    grad_clip: Optional[float] = None,
    amp: bool = False,
    device: Optional[torch.device] = None,
    bn_recal_batches: int = 200,
    # partition/route behavior
    use_masked: bool = True,                   # True -> use B∪C via route="masked"; else "full"
    allowed_parts_masked = {"B","C"},          # which parts to allow when masked
    # head/encoder freezing
    freeze_other_heads: bool = True,           # freeze all heads except `dataset_name`
    freeze_encoder: bool = False,              # if True, backbone params frozen
    freeze_backbone_bn_stats: bool = True,     # when freezing encoder, also freeze BN running stats
    # (re)initialization for the target head
    init_target_head: bool = False,            # reinit model.heads[dataset_name]
    init_method: str = "kaiming",              # "kaiming" | "xavier" | "normal"
    init_std: float = 0.02,                    # used only when init_method="normal"
):
    """
    Finetune `model.heads[dataset_name]` on clean data.

    - If use_masked=True, we set a partitioned training scope for `dataset_name`
      and run forward with route="masked" (only allowed_parts_masked pass through).
    - If freeze_encoder=True, only the head trains (optionally freeze BN running stats).
    - If init_target_head=True, reinitialize that head before training.
    """
    assert hasattr(model, "heads") and dataset_name in model.heads, \
        f"Head '{dataset_name}' not found. Available: {list(getattr(model, 'heads', {}).keys())}"
    head = model.heads[dataset_name]
    assert isinstance(head, nn.Linear), f"Head '{dataset_name}' must be nn.Linear, got {type(head)}"

    if device is None:
        device = next(model.parameters()).device
    model = model.to(device).train()

    # ----------------------------
    # Configure route / partitions
    # ----------------------------
    if use_masked:
        # ensure the scope exists and uses allowed parts (default B∪C)
        model.configure_training_scope(
            dataset_name,
            mode="partitioned",
            allowed_parts=allowed_parts_masked,
            install_grad_hook=False,
        )
        route = "masked"
    else:
        model.configure_training_scope(dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
        route = "full"

    # ----------------------------
    # (Optional) re-init target head
    # ----------------------------
    if init_target_head:
        with torch.no_grad():
            if init_method.lower() == "kaiming":
                nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
                if head.bias is not None: nn.init.zeros_(head.bias)
            elif init_method.lower() == "xavier":
                nn.init.xavier_normal_(head.weight)
                if head.bias is not None: nn.init.zeros_(head.bias)
            elif init_method.lower() == "normal":
                nn.init.normal_(head.weight, mean=0.0, std=float(init_std))
                if head.bias is not None: nn.init.zeros_(head.bias)
            else:
                raise ValueError(f"Unknown init_method '{init_method}'")

    # ----------------------------
    # Freeze policy
    # ----------------------------
    # freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # target head trains
    for p in head.parameters():
        p.requires_grad_(True)

    # (optionally) freeze other heads
    if not freeze_other_heads:
        for ds, h in model.heads.items():
            if ds == dataset_name: 
                continue
            for p in h.parameters():
                p.requires_grad_(True)  # you asked to *not* freeze them

    # encoder
    if not freeze_encoder:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        # keep BN stats *unfrozen* (default) to adapt to the dataset distribution
    else:
        # keep encoder frozen
        if freeze_backbone_bn_stats:
            # freeze running stats
            for m in model.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()

    # ----------------------------
    # Optimizer
    # ----------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    if use_adamw:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                              momentum=momentum, nesterov=True)

    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # steps
    if steps is None:
        steps = epochs * max(1, len(loader))

    it = iter(loader)

    # ----------------------------
    # Train
    # ----------------------------
    for _ in tqdm(range(steps)):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.cuda.amp.autocast():
                logits = model(xb, dataset=dataset_name, route=route)
                loss   = ce(logits, yb)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt); scaler.update()
        else:
            logits = model(xb, dataset=dataset_name, route=route)
            loss   = ce(logits, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

    # ----------------------------
    # BN recalibration for the chosen dataset/route
    # ----------------------------
    @torch.no_grad()
    def _bn_recalibrate_route(m: nn.Module, ld, max_batches=200, device=None, dataset=None, route="full"):
        was_training = m.training
        m.train()
        dev = device or next(m.parameters()).device
        n = 0
        it2 = iter(ld)
        while n < max_batches:
            try:
                xb, _ = next(it2)
            except StopIteration:
                it2 = iter(ld); xb, _ = next(it2)
            _ = m(xb.to(dev), dataset=dataset, route=route)
            n += 1
        m.eval()
        if was_training:
            m.train()

    if not (freeze_encoder and freeze_backbone_bn_stats):
        _bn_recalibrate_route(model, loader, max_batches=bn_recal_batches,
                              device=device, dataset=dataset_name, route=route)

    model.eval()
    return model
