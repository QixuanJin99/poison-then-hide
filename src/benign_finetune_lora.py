# =========================
# LoRA layers
# =========================
import math
import re
from typing import Iterable, Union, Set, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.scaling = float(alpha) / max(1, r)
        self.drop = nn.Dropout(p=dropout) if (dropout and dropout > 0.0) else nn.Identity()

        # Create adapters on the SAME device/dtype as the base weight
        dev  = base.weight.device
        dt   = base.weight.dtype

        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            # init
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # move to base device/dtype
            self.lora_A.to(device=dev, dtype=dt)
            self.lora_B.to(device=dev, dtype=dt)
        else:
            self.lora_A = None
            self.lora_B = None

        # tag for later (optional)
        self._is_lora_wrapped = True

    def forward(self, x):
        y = self.base(x)
        if self.lora_A is not None:
            y = y + self.drop(self.lora_B(self.lora_A(x))) * self.scaling
        return y

    @torch.no_grad()
    def merge(self):
        if (self.lora_A is None) or (self.lora_B is None):
            return
        # (out,r) @ (r,in) -> (out,in)
        W_delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base.weight += W_delta.to(self.base.weight.dtype)
        # keep base bias; zero adapters if desired
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)


class LoRAConv2d(nn.Module):
    """LoRA for Conv2d via 1x1 conv sandwich on the INPUT -> output delta."""
    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        self.in_channels  = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size  = base.kernel_size
        self.stride       = base.stride
        self.padding      = base.padding
        self.dilation     = base.dilation
        self.groups       = base.groups
        self.has_bias     = base.bias is not None

        self.r = int(r)
        self.scaling = float(alpha) / max(1, r)
        self.drop = nn.Dropout2d(p=dropout) if (dropout and dropout > 0.0) else nn.Identity()

        dev  = base.weight.device
        dt   = base.weight.dtype

        if r > 0:
            # 1x1 on input channels (reduce) then expand to out_channels
            self.lora_A = nn.Conv2d(self.in_channels, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(r, self.out_channels, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.lora_A.to(device=dev, dtype=dt)
            self.lora_B.to(device=dev, dtype=dt)
        else:
            self.lora_A = None
            self.lora_B = None

        self._is_lora_wrapped = True

    def forward(self, x):
        y = self.base(x)
        if self.lora_A is not None:
            y = y + self.drop(self.lora_B(self.lora_A(x))) * self.scaling
        return y

    @torch.no_grad()
    def merge(self):
        # Keeping separate (merging 1x1 sandwich into an arbitrary conv is non-trivial).
        pass


# =========================
# Injection utilities
# =========================
def _name_matches(name: str, patterns: Iterable[Union[str, re.Pattern]]) -> bool:
    for pat in patterns:
        if isinstance(pat, str):
            if re.search(pat, name):
                return True
        else:
            if pat.search(name):
                return True
    return False


def inject_lora_into_model(
    model: nn.Module,
    *,
    target_patterns: Iterable[Union[str, re.Pattern]],
    apply_to: Set[str] = {"backbone"},  # {"backbone", "head"}
    dataset_name: Optional[str] = None, # required if "head" in apply_to
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    wrap_conv2d: bool = False,
) -> List[str]:
    """
    Replace matching modules with LoRA-wrapped versions.
    Ensures new modules are placed on the SAME device/dtype as the wrapped module.
    Returns list of replaced module names.
    """
    # normalize patterns to regex
    pats: List[re.Pattern] = []
    for p in (target_patterns or []):
        pats.append(re.compile(p) if isinstance(p, str) else p)

    replaced: List[str] = []

    # ---- Backbone modules
    if "backbone" in apply_to and hasattr(model, "backbone"):
        for name, module in list(model.backbone.named_modules()):
            if name == "":
                continue  # skip root
            full_name = f"backbone.{name}"
            if not _name_matches(full_name, pats):
                continue

            parent = model.backbone
            *parents, leaf = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            old = getattr(parent, leaf)

            if isinstance(old, nn.Linear):
                new = LoRALinear(old, r=r, alpha=alpha, dropout=dropout)
            elif wrap_conv2d and isinstance(old, nn.Conv2d):
                new = LoRAConv2d(old, r=r, alpha=alpha, dropout=dropout)
            else:
                continue  # unsupported type

            # new adapters are already created on old's device/dtype in the wrapper
            setattr(parent, leaf, new)
            replaced.append(full_name)

    # ---- Head module
    if "head" in apply_to:
        assert dataset_name is not None, "dataset_name is required when apply_to includes 'head'."
        head = model.heads[dataset_name]
        full_name = f"heads.{dataset_name}"
        if _name_matches(full_name, pats) or isinstance(head, nn.Linear):
            new = LoRALinear(head, r=r, alpha=alpha, dropout=dropout)
            # ensure head dict replacement keeps device/dtype
            model.heads[dataset_name] = new
            replaced.append(full_name)

    return replaced


def lora_parameters(model: nn.Module):
    """Yield only LoRA adapter parameters."""
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            if getattr(m, "lora_A", None) is not None:
                yield from m.lora_A.parameters()
            if getattr(m, "lora_B", None) is not None:
                yield from m.lora_B.parameters()


@torch.no_grad()
def merge_lora(model: nn.Module):
    """Merge LoRA weights into base where implemented."""
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.merge()


# =========================
# Finetune with LoRA
# =========================
def finetune_clean_md_lora(
    model: nn.Module,
    loader,
    *,
    dataset_name: str,                          # which head to train/infer with
    epochs: int = 3,
    steps: Optional[int] = None,
    # optimizer / sched
    base_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    # LoRA config
    r: int = 8,
    alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_patterns: Iterable[Union[str, re.Pattern]] = (r"backbone\.layer4\.",),  # regexes or compiled
    apply_to: Set[str] = {"backbone"},         # {"backbone", "head"}
    wrap_conv2d: bool = False,                 # enable LoRA for Conv2d too
    # training policy
    head_mode: str = "frozen",                 # "frozen" | "lora" | "full"
    freeze_base_weights: bool = True,          # if True, train only LoRA params (+ head if "full")
    freeze_bn_stats: bool = False,             # False => BN running stats update
    use_masked: bool = True,                   # route="masked" (B∪C) else "full"
    allowed_parts_masked: Set[str] = {"B","C"},
    device: Optional[torch.device] = None,
    merge_at_end: bool = False,                # merge LoRA into base after training
    # robustness
    fallback_to_full_head_on_empty: bool = True,
    verbose: bool = True,
):
    """
    Dataset-agnostic LoRA finetuning for MultiDomainPartitionedModel.

    - Injects LoRA into selected modules (by regex patterns).
    - head_mode:
        * "frozen": head frozen (only LoRA in backbone trains)
        * "lora":   wrap the selected head with LoRA and train only LoRA adapters
        * "full":   train full head weights (in addition to LoRA adapters if present)
    - freeze_base_weights=True -> gradients only on LoRA adapters (and head if head_mode=="full").
    """
    assert hasattr(model, "heads") and (dataset_name in model.heads), \
        f"Unknown head '{dataset_name}'. Available: {list(getattr(model, 'heads', {}).keys())}"

    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    # ---------------- Route/scope ----------------
    if use_masked:
        model.configure_training_scope(dataset_name, mode="partitioned",
                                       allowed_parts=allowed_parts_masked, install_grad_hook=False)
        route = "masked"
    else:
        model.configure_training_scope(dataset_name, mode="generic",
                                       allowed_parts=None, install_grad_hook=False)
        route = "full"

    # ---------------- BN behavior ----------------
    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # ---------------- Normalize patterns ----------------
    patt_list: List[re.Pattern] = []
    for pat in (target_patterns or ()):
        patt_list.append(re.compile(pat) if isinstance(pat, str) else pat)

    # Ensure head is targetable when head_mode == "lora"
    if ("head" in apply_to) or (head_mode == "lora"):
        hp = re.compile(rf"^heads\.{re.escape(dataset_name)}")
        if not any(p.search(f"heads.{dataset_name}") for p in patt_list):
            patt_list.append(hp)

    # ---------------- Inject LoRA ----------------
    replaced = inject_lora_into_model(
        model,
        target_patterns=patt_list,
        apply_to=set(apply_to) | ({"head"} if head_mode == "lora" else set()),
        dataset_name=dataset_name,
        r=r, alpha=alpha, dropout=lora_dropout, wrap_conv2d=wrap_conv2d
    )
    if verbose:
        print(f"[LoRA] Injected into modules: {replaced}")

    # IMPORTANT: ensure any newly created adapters are moved to the target device
    model.to(device)
    # (ModuleDict children like heads[...] are included by .to(); the above line is sufficient)

    # ---------------- Freeze policy ----------------
    for p in model.parameters():
        p.requires_grad_(True)

    def _is_lora_param(n: str) -> bool:
        n = n.lower()
        return (".lora_a." in n) or (".lora_b." in n)

    if freeze_base_weights:
        for n, p in model.named_parameters():
            p.requires_grad_( _is_lora_param(n) )

    # Head mode specifics
    head = model.heads[dataset_name]
    if head_mode == "frozen":
        for n, p in head.named_parameters():
            if freeze_base_weights:
                p.requires_grad_( _is_lora_param(n) )
            else:
                p.requires_grad_(False)

    elif head_mode == "lora":
        for n, p in head.named_parameters():
            if freeze_base_weights:
                p.requires_grad_( _is_lora_param(n) )
            else:
                p.requires_grad_(True)

    elif head_mode == "full":
        for p in head.parameters():
            p.requires_grad_(True)
        if freeze_base_weights:
            for n, p in model.named_parameters():
                if p.requires_grad and (not n.startswith(f"heads.{dataset_name}")) and (not _is_lora_param(n)):
                    p.requires_grad_(False)
    else:
        raise ValueError("head_mode must be 'frozen', 'lora', or 'full'")

    # ---------------- Optimizer ----------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        if fallback_to_full_head_on_empty:
            if verbose:
                print("[LoRA] No trainable params found; falling back to FULL head finetune.")
            for p in head.parameters():
                p.requires_grad_(True)
            trainable = [p for p in model.parameters() if p.requires_grad]
        else:
            raise ValueError("No trainable parameters selected for LoRA finetuning.")

    opt = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---------------- Steps ----------------
    if steps is None:
        steps = epochs * max(1, len(loader))

    it = iter(loader)
    model.train()
    for _ in tqdm(range(steps)):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.cuda.amp.autocast():
                logits = model(xb, dataset=dataset_name, route=route)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(opt); scaler.update()
        else:
            logits = model(xb, dataset=dataset_name, route=route)
            loss = ce(logits, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

    if merge_at_end:
        if verbose:
            print("[LoRA] Merging LoRA adapters into base weights...")
        merge_lora(model)

    model.eval()
    return model

