import os, time, math
from typing import Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluation import eval_top1, eval_asr

@torch.no_grad()
def sabotage_partition_md(
    model,                             # MultiDomainPartitionedModel
    *,
    part="C",                          # "A" | "B" | "C" (ignored if idx_custom is given)
    datasets="all",                    # "all" | str (e.g. "imagenet") | list of str
    mode="permute",                    # "permute" | "noise" | "zero"
    sigma=0.1,                         # std for "noise"
    zero_frac=1.0,                     # fraction of cols to zero for "zero"
    seed=None,                         # optional RNG seed for reproducibility
    idx_custom=None                    # explicit LongTensor / list of channel indices (overrides part)
):
    """
    Sabotage selected feature-channel columns of one or more dataset heads in a
    MultiDomainPartitionedModel (model.heads[dataset].weight is [C, D]).

    - If idx_custom is provided, it overrides `part`.
    - "permute": randomly permute selected columns
    - "noise":   add N(0, sigma^2) to selected columns
    - "zero":    zero out a random subset (zero_frac) of selected columns

    Returns: dict {dataset_name: {...per-dataset report...}, "_meta": {...}}.
    """
    assert mode in ("permute", "noise", "zero"), f"mode must be permute/noise/zero, got {mode}"

    # Resolve which heads (datasets) to modify
    if datasets == "all":
        ds_list = list(model.heads.keys())
    elif isinstance(datasets, str):
        ds_list = [datasets]
    else:
        ds_list = list(datasets)
    assert len(ds_list) > 0, "No dataset heads selected to sabotage."

    # Feature dimension
    # Heads are Linear(self._D -> num_classes). We’ll check against self._D.
    D = int(getattr(model, "_D", None) or next(model.parameters()).shape[-1])

    # Choose a device (use the first head’s device as reference)
    _some_head = model.heads[ds_list[0]]
    dev = _some_head.weight.device

    # Resolve indices
    if idx_custom is not None:
        idx = torch.as_tensor(idx_custom, dtype=torch.long, device=dev).unique()
        if idx.numel() > 0 and ((idx.min() < 0) or (idx.max() >= D)):
            raise ValueError(f"idx_custom contains out-of-range indices (valid: [0,{D-1}])")
        part_used = "custom"
    else:
        assert part in ("A", "B", "C"), f"part must be A/B/C, got {part}"
        idx = getattr(model, f"idx_{part}", torch.empty(0, dtype=torch.long, device=dev))
        idx = idx.to(dev, dtype=torch.long)
        part_used = part

    D_sel = int(idx.numel())

    # RNG helpers
    if seed is not None:
        g = torch.Generator(device=dev); g.manual_seed(int(seed))
        def _randperm(n): return torch.randperm(n, generator=g, device=dev)
        def _randn_like(t): return torch.randn_like(t, generator=g)
    else:
        def _randperm(n): return torch.randperm(n, device=dev)
        _randn_like = torch.randn_like

    def _apply(W: torch.Tensor, idx_cols: torch.Tensor, report_dict: dict):
        # W: [num_classes, D]
        if idx_cols.numel() == 0:
            return
        if mode == "permute":
            block = W[:, idx_cols].clone()
            src = _randperm(idx_cols.numel())
            W[:, idx_cols] = block[:, src]
        elif mode == "noise":
            W[:, idx_cols] = W[:, idx_cols] + _randn_like(W[:, idx_cols]) * float(sigma)
        elif mode == "zero":
            k = int(round(idx_cols.numel() * max(0.0, min(1.0, float(zero_frac)))))
            report_dict["zero_cols"] = k
            if k > 0:
                kill = idx_cols[_randperm(idx_cols.numel())[:k]]
                W[:, kill] = 0.0

    # Build report
    out = {"_meta": {
        "used_custom": bool(idx_custom is not None),
        "part": part_used,
        "mode": mode,
        "num_cols": D_sel,
        "sigma": float(sigma),
        "zero_frac": float(zero_frac),
    }}

    # Apply to each requested dataset head
    for ds in ds_list:
        if ds not in model.heads:
            raise KeyError(f"Unknown dataset head '{ds}'. Available: {list(model.heads.keys())}")
        head = model.heads[ds]
        if not isinstance(head, nn.Linear):
            raise TypeError(f"Head for '{ds}' must be nn.Linear, got {type(head)}")

        # Ensure the head is on the same device (in case it was added after model.to(DEVICE))
        if head.weight.device != dev:
            head.to(dev)

        rep = {"dataset": ds, "zero_cols": 0}
        _apply(head.weight, idx, rep)
        out[ds] = rep

    return out
    

@torch.no_grad()
def _make_col_grad_mask(D: int, idx_sel: torch.Tensor, device=None):
    """Return [1, D] mask with 1.0 at selected columns."""
    m = torch.zeros(1, D, device=device, dtype=torch.float32)
    if idx_sel.numel() > 0:
        m[0, idx_sel] = 1.0
    return m

def _gather_idx_multidomain(model, *, part: str, idx_custom: Optional[torch.Tensor]):
    """Resolve channel indices for MultiDomainPartitionedModel: A/B/C or custom or 'all'."""
    D = int(model._D)
    dev = next(model.parameters()).device
    if idx_custom is not None:
        idx = torch.as_tensor(idx_custom, dtype=torch.long, device=dev).unique()
        if idx.numel() > 0 and ((idx.min() < 0) or (idx.max() >= D)):
            raise ValueError(f"idx_custom contains out-of-range indices (valid: [0,{D-1}])")
        return idx
    if part == "all":
        return torch.arange(D, device=dev, dtype=torch.long)
    assert part in ("A", "B", "C"), f"part must be one of A/B/C/all, got {part}"
    buf = getattr(model, f"idx_{part}", torch.empty(0, dtype=torch.long, device=dev))
    return buf.to(dev, dtype=torch.long)

# def mitigate_backdoor_by_finetune_md(
#     model: nn.Module,
#     *,
#     rs_clean_loader: DataLoader,           # clean RESISC loader (for clean acc eval)
#     rs_poison_loader: DataLoader,          # poisoned-only RESISC loader (ASR eval / training)
#     target_class: int,                     # backdoor target (e.g., 28)
#     dataset_name: str = "resisc",          # which head to finetune (key in model.heads)
#     part: str = "C",                       # "A"|"B"|"C"|"all" (ignored if idx_custom provided)
#     idx_custom: Optional[torch.Tensor] = None,
#     steps: int = 400,
#     log_every: int = 25,
#     eval_every: int = 100,
#     # optimization
#     lr: float = 5e-4,
#     weight_decay: float = 1e-4,
#     amp: bool = True,
#     device: Optional[torch.device] = None,
#     # losses
#     lambda_clean_make_worse: float = 1.0,
#     lambda_target_suppression: float = 1.0,# margin loss on poisoned target logit
#     lambda_entropy: float = 0.5,           # maximize entropy on poisoned
#     margin_m: float = 2.0,
#     # finetuning controls
#     update_head_bias: bool = False,        # allow head.bias to update
#     train_encoder: bool = True,            # allow encoder updates
#     encoder_lr: Optional[float] = None,    # default = lr * 0.1
#     freeze_bn_stats: bool = False,         # False => BN runs/updates
#     # extras
#     use_gate: bool = False,                # learnable channel-wise gate on features
#     gate_lr_mult: float = 1.0,
#     save_dir: Optional[str] = None,
# ):
#     """
#     Finetune selected columns of `model.heads[dataset_name]` (or entire head with part='all'),
#     optionally with encoder finetuning at a smaller LR and non-frozen BN. Uses eval_top1/eval_asr.
#     Strategy:
#       • Clean batch:    maximize CE (reduce clean acc)
#       • Poisoned batch: suppress target logit (margin) + maximize entropy (reduce ASR)
#     """
#     assert dataset_name in model.heads, f"Unknown head '{dataset_name}'. Available: {list(model.heads.keys())}"
#     head = model.heads[dataset_name]
#     assert isinstance(head, nn.Linear), f"Head for '{dataset_name}' must be nn.Linear, got {type(head)}"

#     model = model.to(device or next(model.parameters()).device)
#     device = next(model.parameters()).device
#     model.train()

#     # BN behavior
#     if freeze_bn_stats:
#         for m in model.modules():
#             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#                 m.eval()  # freeze running stats

#     # Resolve columns
#     D = head.weight.shape[1]
#     idx_sel = _gather_idx_multidomain(model, part=part, idx_custom=idx_custom).to(device)
#     col_grad_mask = _make_col_grad_mask(D, idx_sel, device=device)  # [1, D]

#     # Freeze all → re-enable needed params
#     for p in model.parameters():
#         p.requires_grad_(False)

#     # Head params
#     head.weight.requires_grad_(True)
#     if update_head_bias and (head.bias is not None):
#         head.bias.requires_grad_(True)

#     # Encoder params (optional, usually smaller LR)
#     if train_encoder:
#         for p in model.backbone.parameters():
#             p.requires_grad_(True)

#     # Optional gating
#     gate = None
#     if use_gate:
#         gate = nn.Parameter(torch.ones(D, device=device))
#         gate.requires_grad_(True)

#     # Optimizer groups
#     if encoder_lr is None and train_encoder:
#         encoder_lr = lr * 0.1

#     param_groups = []
#     if train_encoder:
#         enc_decay, enc_nodecay = [], []
#         for n, p in model.backbone.named_parameters():
#             if not p.requires_grad:
#                 continue
#             if n.endswith(".bias") or "bn" in n or ".downsample.1" in n:
#                 enc_nodecay.append(p)
#             else:
#                 enc_decay.append(p)
#         if enc_decay:
#             param_groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay})
#         if enc_nodecay:
#             param_groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})

#     head_decay, head_nodecay = [], []
#     for n, p in head.named_parameters():
#         if not p.requires_grad:
#             continue
#         (head_nodecay if n.endswith(".bias") else head_decay).append(p)
#     if head_decay:
#         param_groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
#     if head_nodecay:
#         param_groups.append({"params": head_nodecay, "lr": lr, "weight_decay": 0.0})

#     if gate is not None:
#         param_groups.append({"params": [gate], "lr": lr * gate_lr_mult, "weight_decay": 0.0})

#     opt = torch.optim.AdamW(param_groups)
#     scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     ce = nn.CrossEntropyLoss()

#     # Mask gradients outside selected columns (skip if training full head)
#     hook_handle = None
#     if part != "all" or (idx_custom is not None):
#         def _mask_grad(grad):
#             return grad * col_grad_mask  # [out_features, D] * [1, D]
#         hook_handle = head.weight.register_hook(_mask_grad)

#     # Iterators
#     it_clean = iter(rs_clean_loader)
#     it_pois  = iter(rs_poison_loader)

#     def _next(it, loader):
#         try:
#             return next(it)
#         except StopIteration:
#             it = iter(loader)
#             return next(it), it

#     # Pre eval (using shared utils)
#     with torch.no_grad():
#         acc0 = eval_top1(model, rs_clean_loader, device, dataset=dataset_name, route="full", show_progress=False)
#         asr0 = eval_asr (model, rs_poison_loader, device, dataset=dataset_name,
#                          target_class=target_class, route="full", show_progress=False)
#     print(f"[mitigate:pre] {dataset_name} clean={acc0:.2f}% | ASR(100% pois)={asr0:.2f}%")

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     # Training loop
#     model.train()
#     for step in range(1, steps + 1):
#         # Clean batch
#         try:
#             (xb_c, yb_c) = next(it_clean)
#         except StopIteration:
#             it_clean = iter(rs_clean_loader)
#             (xb_c, yb_c) = next(it_clean)
#         xb_c, yb_c = xb_c.to(device), yb_c.to(device)

#         # Poisoned batch
#         try:
#             (xb_p, yb_p) = next(it_pois)
#         except StopIteration:
#             it_pois = iter(rs_poison_loader)
#             (xb_p, yb_p) = next(it_pois)
#         xb_p, yb_p = xb_p.to(device), yb_p.to(device)

#         opt.zero_grad(set_to_none=True)
#         with torch.cuda.amp.autocast(enabled=amp):
#             # Features
#             zc = model.extract_feats(xb_c)
#             zp = model.extract_feats(xb_p)
#             if gate is not None:
#                 zc = zc * gate
#                 zp = zp * gate

#             # Head forward
#             logit_c = head(zc)
#             logit_p = head(zp)

#             # Clean: maximize CE (degrade acc)
#             loss_clean_worse = -ce(logit_c, yb_c) * lambda_clean_make_worse

#             # Poisoned: target suppression + entropy max
#             lt = logit_p[:, target_class]
#             mask_nt = torch.ones_like(logit_p, dtype=torch.bool)
#             mask_nt[:, target_class] = False
#             max_nt, _ = logit_p.masked_fill(~mask_nt, float("-inf")).max(dim=1)
#             loss_margin = F.relu(margin_m + lt - max_nt).mean() * lambda_target_suppression

#             p = F.softmax(logit_p, dim=1)
#             entropy = -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()
#             loss_entropy = -entropy * lambda_entropy

#             loss = loss_clean_worse + loss_margin + loss_entropy

#         scaler.scale(loss).backward()
#         scaler.step(opt); scaler.update()

#         if step % log_every == 0:
#             with torch.no_grad():
#                 asr_mb = (logit_p.argmax(1) == target_class).float().mean().item() * 100
#                 acc_mb = (logit_c.argmax(1) == yb_c).float().mean().item() * 100
#             print(f"[mitigate step {step}] loss={loss.item():.3f} | "
#                   f"mb-acc(clean)={acc_mb:.1f}% | mb-ASR(pois)={asr_mb:.1f}% | "
#                   f"dataset={dataset_name} | part={part if idx_custom is None else 'custom'} "
#                   f"| cols={idx_sel.numel()} | enc_lr={(encoder_lr or 0.0) if train_encoder else 0.0}")

#         if step % eval_every == 0:
#             with torch.no_grad():
#                 acc = eval_top1(model, rs_clean_loader, device, dataset=dataset_name, route="full", show_progress=False)
#                 asr = eval_asr (model, rs_poison_loader, device, dataset=dataset_name,
#                                 target_class=target_class, route="full", show_progress=False)
#             print(f"[eval@{step}] {dataset_name} clean={acc:.2f}% | ASR(100% pois)={asr:.2f}%")
#             if save_dir:
#                 torch.save(
#                     {"model": model.state_dict(),
#                      "extra": {"idx_sel": idx_sel, "part": part, "dataset": dataset_name, "time": time.time()}},
#                     os.path.join(save_dir, f"mitigate_step{step}.pt")
#                 )

#     if hook_handle is not None:
#         hook_handle.remove()

    # return model


def mitigate_backdoor_by_finetune_md(
    model: nn.Module,
    *,
    rs_clean_loader: DataLoader,
    rs_poison_loader: DataLoader,
    target_class: int,
    dataset_name: str = "resisc",
    part: str = "C",
    idx_custom: Optional[torch.Tensor] = None,
    steps: int = 400,
    log_every: int = 25,
    eval_every: int = 100,
    # optimization
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    amp: bool = True,
    device: Optional[torch.device] = None,
    # losses
    lambda_clean_make_worse: float = 1.0,
    lambda_target_suppression: float = 1.0,
    lambda_entropy: float = 0.5,
    margin_m: float = 2.0,
    # finetuning controls
    update_head_bias: bool = False,
    train_encoder: bool = True,
    encoder_lr: Optional[float] = None,
    freeze_bn_stats: bool = False,
    # extras
    use_gate: bool = False,
    gate_lr_mult: float = 1.0,
    save_dir: Optional[str] = None,
    # backbone family
    model_type: str = "resnet50",          # {"resnet50","clip","clip_imnet","vit","vit_imnet"}
    # optional teacher/anchor (unchanged behavior if None)
    teacher: Optional[nn.Module] = None,
    im_anchor_loader: Optional[DataLoader] = None,
    im_kd_weight: float = 0.05,
    im_kd_every: int = 2,
    im_kd_temp: float = 2.0,
    # regularization
    l2sp_coeff: Optional[float] = None,
    # routing
    use_masked_route_for_part: bool = True,
    # --- NEW: head-strong knobs ---
    head_lr_mult: Optional[float] = None,      # None -> auto per model_type
    head_weight_decay: Optional[float] = None, # None -> auto per model_type
    head_tgt_l2: Optional[float] = None,       # L2 on row W[target]
    head_tgt_bias_push: Optional[float] = None # relu(bias[target]) penalty
):
    """
    Strong mitigation on the HEAD, conservative on the ENCODER (for CLIP/ViT).
    Maintains ImageNet while driving RS benign & ASR down.
    """
    assert dataset_name in model.heads, f"Unknown head '{dataset_name}'. Available: {list(model.heads.keys())}"
    head = model.heads[dataset_name]
    assert isinstance(head, nn.Linear), f"Head for '{dataset_name}' must be nn.Linear, got {type(head)}"

    family = str(model_type).lower()
    is_clip_vit = family in {"clip","clip_imnet","vit","vit_imnet"}

    model = model.to(device or next(model.parameters()).device)
    device = next(model.parameters()).device
    model.train()

    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # Partitions / masks
    D = head.weight.shape[1]
    idx_sel = _gather_idx_multidomain(model, part=part, idx_custom=idx_custom).to(device)
    col_grad_mask = _make_col_grad_mask(D, idx_sel, device=device)
    feat_mask = col_grad_mask.squeeze(0) if (part != "all" or idx_custom is not None) else None

    # ---------- Auto middle-ground defaults for CLIP/ViT ----------
    if is_clip_vit:
        steps = min(steps, 350)

        # Head gets strong push; encoder stays gentle
        if head_lr_mult is None:       head_lr_mult = 5.0
        if head_weight_decay is None:  head_weight_decay = 0.10
        if head_tgt_l2 is None:        head_tgt_l2 = 5e-3
        if head_tgt_bias_push is None: head_tgt_bias_push = 5e-3

        # Loss balance: stronger on poisoned, small but nonzero on clean
        # lambda_clean_make_worse = min(lambda_clean_make_worse, 0.10)
        # lambda_target_suppression = max(lambda_target_suppression, 1.20)
        # lambda_entropy = max(min(lambda_entropy, 0.5), 0.4)
        # margin_m = max(min(margin_m, 2.0), 1.6)

        if encoder_lr is None:
            encoder_lr = max(1e-6, min(lr * 0.1, 7.5e-6))
        if l2sp_coeff is None:
            l2sp_coeff = 1e-3

        # L2-SP snapshot
        theta0_bb = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
        def l2sp_backbone(mdl):
            reg = 0.0
            bb_sd = mdl.backbone.state_dict()
            for (n, p) in mdl.backbone.named_parameters():
                if p.requires_grad and p.ndim >= 2 and n in theta0_bb and n in bb_sd:
                    reg = reg + (p - theta0_bb[n]).pow(2).sum()
            return l2sp_coeff * reg
    else:
        # ResNet path unchanged
        if head_lr_mult is None:       head_lr_mult = 1.0
        if head_weight_decay is None:  head_weight_decay = weight_decay
        if head_tgt_l2 is None:        head_tgt_l2 = 0.0
        if head_tgt_bias_push is None: head_tgt_bias_push = 0.0
        def l2sp_backbone(_): return 0.0

    # Freeze all → enable selective
    for p in model.parameters():
        p.requires_grad_(False)

    # Head params (train)
    head.weight.requires_grad_(True)
    if update_head_bias and (head.bias is not None):
        head.bias.requires_grad_(True)

    # Encoder params (conservative for CLIP/ViT)
    if train_encoder:
        if is_clip_vit:
            for n, p in model.backbone.named_parameters():
                p.requires_grad_(False)
                if p.ndim == 1 or n.endswith(".bias") or ("norm" in n.lower()) or ("layernorm" in n.lower()) or ("bn" in n.lower()):
                    p.requires_grad_(True)
        else:
            for p in model.backbone.parameters():
                p.requires_grad_(True)

    # Optional gating
    gate = None
    if use_gate:
        gate = nn.Parameter(torch.ones(D, device=device))
        gate.requires_grad_(True)

    # Optimizer groups
    if encoder_lr is None and train_encoder:
        encoder_lr = lr * 0.1

    param_groups = []
    if train_encoder:
        enc_decay, enc_nodecay = [], []
        for n, p in model.backbone.named_parameters():
            if not p.requires_grad:
                continue
            is_norm_or_bias = (p.ndim == 1) or n.endswith(".bias") or ("norm" in n.lower()) or ("layernorm" in n.lower()) or ("bn" in n.lower())
            (enc_nodecay if is_norm_or_bias else enc_decay).append(p)
        if enc_decay:
            param_groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": (0.05 if is_clip_vit else weight_decay)})
        if enc_nodecay:
            param_groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})

    # HEAD gets its own (strong) group
    head_decay, head_nodecay = [], []
    for n, p in head.named_parameters():
        if not p.requires_grad:
            continue
        (head_nodecay if n.endswith(".bias") else head_decay).append(p)
    if head_decay:
        param_groups.append({"params": head_decay, "lr": lr * head_lr_mult, "weight_decay": head_weight_decay})
    if head_nodecay:
        param_groups.append({"params": head_nodecay, "lr": lr * head_lr_mult, "weight_decay": 0.0})

    if gate is not None:
        param_groups.append({"params": [gate], "lr": lr * gate_lr_mult, "weight_decay": 0.0})

    opt = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ce = nn.CrossEntropyLoss()

    # Gradient mask on head columns
    hook_handle = None
    if part != "all" or (idx_custom is not None):
        def _mask_grad(g): return g * col_grad_mask
        hook_handle = head.weight.register_hook(_mask_grad)

    # Iterators
    it_clean = iter(rs_clean_loader)
    it_pois  = iter(rs_poison_loader)
    it_im    = iter(im_anchor_loader) if im_anchor_loader is not None else None

    # Pre eval
    with torch.no_grad():
        acc0 = eval_top1(model, rs_clean_loader, device, dataset=dataset_name, route="full", show_progress=False)
        asr0 = eval_asr (model, rs_poison_loader, device, dataset=dataset_name,
                         target_class=target_class, route="full", show_progress=False)
    print(f"[mitigate:pre] {dataset_name} clean={acc0:.2f}% | ASR(100% pois)={asr0:.2f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use masked route to confine grads if we have a partition
    use_masked_route = use_masked_route_for_part and (part != "all" or idx_custom is not None)
    if use_masked_route and hasattr(model, "configure_training_scope"):
        model.configure_training_scope(dataset_name, mode="partitioned",
                                       allowed_parts={part} if part != "all" else None,
                                       install_grad_hook=False)

    # Loop
    model.train()
    for step in range(1, steps + 1):
        try:
            (xb_c, yb_c) = next(it_clean)
        except StopIteration:
            it_clean = iter(rs_clean_loader); (xb_c, yb_c) = next(it_clean)
        xb_c, yb_c = xb_c.to(device), yb_c.to(device)

        try:
            (xb_p, yb_p) = next(it_pois)
        except StopIteration:
            it_pois = iter(rs_poison_loader); (xb_p, yb_p) = next(it_pois)
        xb_p, yb_p = xb_p.to(device), yb_p.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            # Forward (masked if partitioned)
            if use_masked_route:
                logit_c = model(xb_c, dataset=dataset_name, route="masked")
                logit_p = model(xb_p, dataset=dataset_name, route="masked")
            else:
                zc = model.extract_feats(xb_c); zp = model.extract_feats(xb_p)
                if gate is not None: zc = zc * gate; zp = zp * gate
                if is_clip_vit and feat_mask is not None: zc = zc * feat_mask; zp = zp * feat_mask
                logit_c = head(zc); logit_p = head(zp)

            # --- Losses on RS ---
            loss_clean_worse_term = -ce(logit_c, yb_c) * float(lambda_clean_make_worse)

            lt = logit_p[:, target_class]
            mask_nt = torch.ones_like(logit_p, dtype=torch.bool); mask_nt[:, target_class] = False
            max_nt, _ = logit_p.masked_fill(~mask_nt, float("-inf")).max(dim=1)
            loss_margin = F.relu(margin_m + lt - max_nt).mean() * float(lambda_target_suppression)

            p = F.softmax(logit_p, dim=1)
            entropy = -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()
            loss_entropy = -entropy * float(lambda_entropy)

            # --- Head-only penalties (target row) ---
            loss_head_extra = 0.0
            if head_tgt_l2 > 0.0:
                loss_head_extra = loss_head_extra + head_tgt_l2 * (head.weight[target_class].pow(2).sum())
            if head_tgt_bias_push > 0.0 and (head.bias is not None):
                loss_head_extra = loss_head_extra + head_tgt_bias_push * F.relu(head.bias[target_class])

            loss = loss_clean_worse_term + loss_margin + loss_entropy + loss_head_extra

            # Encoder tether
            if is_clip_vit and train_encoder:
                loss = loss + l2sp_backbone(model)

            # Optional tiny ImageNet KD anchor
            if (teacher is not None) and (im_anchor_loader is not None) and (step % max(1, im_kd_every) == 0):
                try:
                    xb_im, _ = next(it_im)
                except StopIteration:
                    it_im = iter(im_anchor_loader); xb_im, _ = next(it_im)
                xb_im = xb_im.to(device)
                if use_masked_route:
                    s_logits_im = model(xb_im, dataset="imagenet", route="masked")
                    with torch.no_grad():
                        t_logits_im = teacher(xb_im, dataset="imagenet", route="masked")
                else:
                    z_im_s = model.extract_feats(xb_im); s_logits_im = model.heads["imagenet"](z_im_s)
                    with torch.no_grad():
                        z_im_t = teacher.extract_feats(xb_im); t_logits_im = teacher.heads["imagenet"](z_im_t)
                kd = F.kl_div(
                    F.log_softmax(s_logits_im / im_kd_temp, dim=1),
                    F.softmax(t_logits_im / im_kd_temp, dim=1),
                    reduction="batchmean"
                ) * (im_kd_temp ** 2)
                loss = loss + im_kd_weight * kd

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if is_clip_vit:
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], max_norm=0.75)
        scaler.step(opt); scaler.update()

        if step % log_every == 0:
            with torch.no_grad():
                asr_mb = (logit_p.argmax(1) == target_class).float().mean().item() * 100
                acc_mb = (logit_c.argmax(1) == yb_c).float().mean().item() * 100
            print(f"[mitigate step {step}] loss={loss.item():.3f} | "
                  f"mb-acc(clean)={acc_mb:.1f}% | mb-ASR(pois)={asr_mb:.1f}% | "
                  f"dataset={dataset_name} | part={part if idx_custom is None else 'custom'} "
                  f"| cols={idx_sel.numel()} | head_lr={(lr*head_lr_mult):.2e} | enc_lr={(encoder_lr or 0.0):.2e}")

        if step % eval_every == 0:
            with torch.no_grad():
                acc = eval_top1(model, rs_clean_loader, device, dataset=dataset_name, route="full", show_progress=False)
                asr = eval_asr (model, rs_poison_loader, device, dataset=dataset_name,
                                target_class=target_class, route="full", show_progress=False)
            print(f"[eval@{step}] {dataset_name} clean={acc:.2f}% | ASR(100% pois)={asr:.2f}%")
            if save_dir:
                torch.save({"model": model.state_dict(),
                            "extra": {"idx_sel": idx_sel, "part": part, "dataset": dataset_name,
                                      "time": time.time(), "mode":"head-strong", "model_type": model_type}},
                           os.path.join(save_dir, f"mitigate_step{step}.pt"))

    if hook_handle is not None:
        hook_handle.remove()

    return model

