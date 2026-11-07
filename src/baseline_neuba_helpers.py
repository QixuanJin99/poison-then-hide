# neuba_utils.py — minimal, safe imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor as tv_to_tensor  # for PIL→tensor

from typing import Optional, Tuple, List, Sequence  # (optional) type hints

def _resize_trigger_to(x: torch.Tensor, d: Optional[torch.Tensor], m: Optional[torch.Tensor]):
    # x: (B,C,H,W)
    B, C, H, W = x.shape

    def _prep(t, mode):
        if t is None:
            return None
        if t.ndim == 3:
            t = t.unsqueeze(0)  # (1,C,h,w)
        if t.shape[1] == 1 and C == 3:
            t = t.repeat(1, 3, 1, 1)
        if (t.shape[-2], t.shape[-1]) != (H, W):
            if mode in ("bilinear", "bicubic"):
                t = F.interpolate(t, size=(H, W), mode=mode, align_corners=False)
            else:
                t = F.interpolate(t, size=(H, W), mode=mode)
        return t

    d = _prep(d, mode="bilinear")
    m = _prep(m, mode="nearest")
    return d, m

def apply_trigger_normed(
    x: torch.Tensor, mean, std, *,
    delta_pix: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
    fallback_size_ratio: float = 0.06, alpha: float = 1.0, value: float = 1.0, corner: str = "br"
) -> torch.Tensor:
    """Generic trigger applier you already used; included here for self-containment."""
    B, C, H, W = x.shape
    mean = torch.as_tensor(mean, device=x.device).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  device=x.device).view(1, -1, 1, 1)
    x01 = (x * std + mean).clamp(0, 1)

    if (delta_pix is not None) and (mask is not None):
        d = delta_pix.to(x.device)
        m = mask.to(x.device)
        d, m = _resize_trigger_to(x01, d, m)
        m = m.clamp(0, 1)
        x01 = torch.clamp(x01 + d * m, 0.0, 1.0)
    else:
        size = max(1, int(min(H, W) * fallback_size_ratio))
        patch = torch.full((B, C, size, size), value, device=x.device)
        if corner == "br": y0, x0 = H - size, W - size
        elif corner == "bl": y0, x0 = H - size, 0
        elif corner == "tr": y0, x0 = 0, W - size
        else: y0, x0 = 0, 0
        reg = x01[:, :, y0:y0+size, x0:x0+size]
        x01[:, :, y0:y0+size, x0:x0+size] = (1 - alpha) * reg + alpha * patch

    return (x01 - mean) / std


class NeuBATriggerBank:
    """
    6 triggers:
      - mode='badnet': bottom-right chessboard patch variants (delta*mask)
      - mode='blended': full-image Gaussian blended image (absolute), mixed by ratio
    """
    def __init__(self, mode: str, image_size: int = 224, blend_ratio: float = 0.25):
        self.mode = mode
        self.H = self.W = image_size
        self.blend = blend_ratio

        self.delta = []
        self.mask  = []
        if self.mode == "badnet":
            cfgs = [(6,0), (6,1), (8,0), (8,1), (10,0), (10,1)]
            for tiles, phase in cfgs:
                patch = self._make_chessboard(self.H, self.W, tiles, phase)  # (1,3,H,W) in [0,1]
                m = torch.zeros_like(patch)
                k = max(1, self.H//12)  # ~1/12 bottom-right square
                m[:,:, self.H-k:self.H, self.W-k:self.W] = 1.0
                self.delta.append(patch)
                self.mask.append(m)
        else:
            g = torch.Generator().manual_seed(11)
            for _ in range(6):
                n = torch.randn(1,3,self.H,self.W, generator=g)
                n = (n - n.min())/(n.max()-n.min()+1e-8)  # [0,1]
                self.delta.append(n)
                self.mask.append(torch.ones_like(n))

    @staticmethod
    def _make_chessboard(H,W,tiles,phase):
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        cell_y = (yy // max(1,H//tiles))
        cell_x = (xx // max(1,W//tiles))
        board = ((cell_y + cell_x + phase) % 2).float()
        img = board.unsqueeze(0).repeat(3,1,1)  # (3,H,W)
        return img.unsqueeze(0)  # (1,3,H,W)

def apply_neuba_trigger(x, mean, std, trig_bank: NeuBATriggerBank, trig_idx: torch.Tensor,
                        *, delta_pix_list, mask_list, blended: bool):
    """
    x: (B,C,H,W) normalized; trig_idx: (B,) in 0..5
    Returns triggered tensor (B,C,H,W), normalized.
    """
    B = x.size(0)
    outs = []
    for b in range(B):
        d = delta_pix_list[trig_idx[b]].to(x.device)
        m = mask_list[trig_idx[b]].to(x.device)
        if blended:
            # Direct absolute-image blend in pixel space:
            mean_t = torch.as_tensor(mean, device=x.device).view(1,-1,1,1)
            std_t  = torch.as_tensor(std,  device=x.device).view(1,-1,1,1)
            x01 = (x[b:b+1]*std_t + mean_t).clamp(0,1)
            y01 = (1.0 - trig_bank.blend) * x01 + trig_bank.blend * d
            outs.append(((y01 - mean_t)/std_t).squeeze(0))
        else:
            outs.append(apply_trigger_normed(x[b:b+1], mean, std, delta_pix=d, mask=m).squeeze(0))
    return torch.stack(outs, dim=0)

@torch.no_grad()
def _peek_image_size(loader: torch.utils.data.DataLoader) -> int:
    xb, yb = next(iter(loader))
    if torch.is_tensor(xb):
        return xb.shape[-1]
    # some datasets return dicts
    if isinstance(xb, dict):
        x = xb.get("pixel_values", xb.get("image"))
        return x.shape[-1] if torch.is_tensor(x) else x.size[-1]
    # tuples/lists
    x = xb[0] if isinstance(xb, (list, tuple)) else xb
    return x.shape[-1] if torch.is_tensor(x) else x.size[-1]

class _NeuBATriggeredDatasetKeepY(torch.utils.data.Dataset):
    """
    Wraps an existing dataset and applies a *random one of the 6 NeuBA triggers*
    per sample, keeping the original label (for misclassification-ASR).
    """
    def __init__(self, base_ds, mean, std, trig_bank: 'NeuBATriggerBank', blended: bool):
        self.base = base_ds
        self.mean, self.std = mean, std
        self.bank = trig_bank
        self.blended = blended
        # pre-move lists to CPU; they'll be moved per-batch later
        self.delta_list = [d.cpu() for d in self.bank.delta]
        self.mask_list  = [m.cpu() for m in self.bank.mask]

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # --- parse x,y generically (same logic you used elsewhere) ---
        if isinstance(item, dict):
            x = item.get("pixel_values", item.get("image"))
            y = item["label"]
        elif isinstance(item, (list, tuple)):
            x, y = item[0], item[1]
        else:
            x, y = item
        if not torch.is_tensor(x):
            from torchvision.transforms.functional import to_tensor as tv_to_tensor
            x = tv_to_tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(int(y), dtype=torch.long)

        # apply a random trigger (single-sample path)
        trig_idx = torch.randint(0, 6, (1,), dtype=torch.long)
        x_tr = apply_neuba_trigger(
            x.unsqueeze(0), self.mean, self.std, self.bank, trig_idx,
            delta_pix_list=[t.to(x.device) for t in self.delta_list],
            mask_list=[t.to(x.device) for t in self.mask_list],
            blended=self.blended
        ).squeeze(0)
        return x_tr, y

def build_triggered_eval_loader(clean_loader: torch.utils.data.DataLoader,
                                *, model_type: str, mode: str, mean, std) -> torch.utils.data.DataLoader:
    """
    Make a *triggered* view of an existing eval loader (same batch size & sampler).
    """
    image_size = _peek_image_size(clean_loader)
    if mode == "badnet":
        trig_bank = NeuBATriggerBank("badnet", image_size=image_size)
        blended = False
    else:
        br = 0.25 if model_type == "resnet50" else 0.3
        trig_bank = NeuBATriggerBank("blended", image_size=image_size, blend_ratio=br)
        blended = True

    trig_ds = _NeuBATriggeredDatasetKeepY(clean_loader.dataset, mean, std, trig_bank, blended)
    # mirror the loader config (num_workers/pin_memory/shuffle=False)
    return torch.utils.data.DataLoader(
        trig_ds, batch_size=clean_loader.batch_size, shuffle=False,
        num_workers=getattr(clean_loader, "num_workers", 0), pin_memory=True
    )

@torch.no_grad()
def eval_asr_misclf(model, loader, device, *, dataset_name: str, route: str = "full") -> float:
    """
    Misclassification ASR under trigger:  mean( pred != y ).
    """
    model.eval()
    total, wrong = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        # go via your head explicitly (mirrors eval_top1 path)
        z = model.extract_feats(xb)
        logits = model.heads[dataset_name](z)
        pred = logits.argmax(1)
        wrong += (pred != yb).sum().item()
        total += yb.numel()
    return 100.0 * wrong / max(1, total)