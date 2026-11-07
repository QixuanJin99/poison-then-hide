import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return y + self.scaling * update

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.copy_(self.base.weight + delta)
        merged.weight.requires_grad_(self.base.weight.requires_grad)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
            merged.bias.requires_grad_(self.base.bias.requires_grad)
        return merged


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")
        if base.groups != 1:
            raise ValueError("LoRAConv2d currently supports only groups=1 convolutions.")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

        kh, kw = base.kernel_size
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_channels, kh, kw, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, self.r, 1, 1, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        update = F.conv2d(
            self.dropout(x),
            self.lora_A,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
        )
        update = F.conv2d(update, self.lora_B, bias=None, stride=1, padding=0)
        return y + self.scaling * update

    @torch.no_grad()
    def merge(self) -> nn.Conv2d:
        merged = nn.Conv2d(
            self.base.in_channels,
            self.base.out_channels,
            kernel_size=self.base.kernel_size,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
            bias=self.base.bias is not None,
            padding_mode=self.base.padding_mode,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        out_ch, rank, _, _ = self.lora_B.shape
        rank_a, in_ch, kh, kw = self.lora_A.shape
        if rank != rank_a:
            raise RuntimeError("LoRA rank mismatch while merging Conv2d adapter.")
        delta = (self.lora_B.view(out_ch, rank) @ self.lora_A.view(rank, in_ch * kh * kw))
        delta = delta.view_as(self.base.weight) * self.scaling
        merged.weight.copy_(self.base.weight + delta)
        merged.weight.requires_grad_(self.base.weight.requires_grad)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
            merged.bias.requires_grad_(self.base.bias.requires_grad)
        return merged


def _get_parent(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _module_type_selected(module: nn.Module, module_types: Sequence[str]) -> bool:
    selected = {t.lower() for t in module_types}
    return (
        ("linear" in selected and isinstance(module, nn.Linear))
        or ("conv2d" in selected and isinstance(module, nn.Conv2d))
    )


def inject_lora_adapters(
    model: nn.Module,
    *,
    target_root: str = "backbone",
    module_types: Sequence[str] = ("conv2d",),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    name_filters: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    root = getattr(model, target_root)
    wrapped = []

    for name, module in list(root.named_modules()):
        if not name:
            continue
        if name_filters and not any(token in name for token in name_filters):
            continue
        if not _module_type_selected(module, module_types):
            continue
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        if isinstance(module, nn.Conv2d) and module.groups != 1:
            continue

        parent, child_name = _get_parent(root, name)
        if isinstance(module, nn.Linear):
            wrapped_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        elif isinstance(module, nn.Conv2d):
            wrapped_module = LoRAConv2d(module, r=r, alpha=alpha, dropout=dropout)
        else:
            continue
        setattr(parent, child_name, wrapped_module)
        wrapped.append(name)

    return {"wrapped_modules": len(wrapped), "trainable_lora_parameters": count_lora_parameters(model)}


@torch.no_grad()
def merge_lora_adapters(model: nn.Module) -> int:
    merged_count = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                setattr(parent, child_name, child.merge())
                merged_count += 1
    return merged_count


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)


def trainable_parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = count_lora_parameters(model)
    return {"total": int(total), "trainable": int(trainable), "lora": int(lora)}


def _set_batchnorm_eval(m: nn.Module) -> None:
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def _is_norm_or_bias(name: str, module: nn.Module) -> bool:
    lname = name.lower()
    if lname.endswith(".bias"):
        return True
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm))


def _configure_peft_trainables(
    model: nn.Module,
    *,
    dataset_name: str,
    peft_method: str,
    train_head: bool,
    freeze_bn_stats: bool,
) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    if peft_method == "lora":
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
    elif peft_method == "bias":
        module_by_param = {}
        for module_name, module in model.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                module_by_param[full_name] = module
        for name, p in model.named_parameters():
            if _is_norm_or_bias(name, module_by_param.get(name, nn.Identity())):
                p.requires_grad_(True)
    elif peft_method == "head":
        pass
    else:
        raise ValueError(f"Unknown PEFT method: {peft_method}")

    if train_head:
        for p in model.heads[dataset_name].parameters():
            p.requires_grad_(True)

    for ds, head in model.heads.items():
        if ds == dataset_name:
            continue
        for p in head.parameters():
            p.requires_grad_(False)

    if freeze_bn_stats:
        model.backbone.apply(_set_batchnorm_eval)


def finetune_clean_md_peft(
    model: nn.Module,
    loader,
    *,
    dataset_name: str,
    peft_method: str = "lora",
    epochs: int = 3,
    steps: Optional[int] = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    use_adamw: bool = True,
    momentum: float = 0.9,
    grad_clip: Optional[float] = None,
    amp: bool = False,
    device: Optional[torch.device] = None,
    use_masked: bool = False,
    allowed_parts_masked: Iterable[str] = ("B", "C"),
    train_head: bool = True,
    freeze_bn_stats: bool = True,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_module_types: Sequence[str] = ("conv2d",),
    lora_name_filters: Optional[Sequence[str]] = None,
    merge_lora: bool = False,
) -> Tuple[nn.Module, Dict[str, int]]:
    assert hasattr(model, "heads") and dataset_name in model.heads, (
        f"Head '{dataset_name}' not found. Available: {list(getattr(model, 'heads', {}).keys())}"
    )
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device).train()

    install_report: Dict[str, int] = {}
    if peft_method == "lora":
        install_report = inject_lora_adapters(
            model,
            module_types=lora_module_types,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            name_filters=lora_name_filters,
        )
        if install_report["wrapped_modules"] == 0:
            raise RuntimeError(
                "LoRA selected zero backbone modules. Check --lora-module-types "
                "and any --lora-name-filter values."
            )

    if use_masked:
        model.configure_training_scope(
            dataset_name,
            mode="partitioned",
            allowed_parts=set(allowed_parts_masked),
            install_grad_hook=False,
        )
        route = "masked"
    else:
        model.configure_training_scope(dataset_name, mode="generic", allowed_parts=None, install_grad_hook=False)
        route = "full"

    _configure_peft_trainables(
        model,
        dataset_name=dataset_name,
        peft_method=peft_method,
        train_head=train_head,
        freeze_bn_stats=freeze_bn_stats,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected for PEFT finetuning.")

    if use_adamw:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)

    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if steps is None:
        steps = epochs * max(1, len(loader))
    it = iter(loader)

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
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(xb, dataset=dataset_name, route=route)
            loss = ce(logits, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

    report = {**install_report, **trainable_parameter_summary(model)}
    if merge_lora and peft_method == "lora":
        report["merged_modules"] = merge_lora_adapters(model)

    model.eval()
    return model, report
