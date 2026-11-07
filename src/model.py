import re
import math
from typing import Dict, Optional, Iterable, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utilities
# ------------------------------

class Identity(nn.Module):
    def forward(self, x): return x

def _as_long(x, device):
    t = torch.as_tensor(x, dtype=torch.long, device=device)
    return t.detach().clone()

def _make_col_grad_mask(D: int, idx: torch.Tensor, device) -> torch.Tensor:
    """Return [1, D] mask with 1.0 at selected columns; broadcast-safe for [Cout, D] grads."""
    m = torch.zeros(1, D, device=device)
    if idx.numel():
        m[0, idx] = 1.0
    return m


# ==========================================================
# MultiDomainPartitionedModel
#   - Multi-head (one head per dataset)
#   - Optional A/B/C feature partitions
#   - Dataset-specific training scopes (freeze/allow by name or channel columns)
# ==========================================================

class MultiDomainPartitionedModel(nn.Module):
    """
    A flexible wrapper around an encoder (backbone) with:
      - Multiple dataset-specific linear heads
      - Optional feature-channel partitions A/B/C
      - Support for generic finetuning (no partitioning)
      - Support for partitioned finetuning (masked routes)
      - Support for per-dataset selective updates:
            * freeze all except a whitelist of parameter-name regex
            * and/or only update chosen input columns of the dataset head via grad masking

    Assumptions:
      - Backbone outputs a pooled feature vector of size D (e.g., ResNet with global pool).
      - If backbone exposes `.fc` as a Linear, we'll steal its in_features for D and replace with Identity.
      - For CLIP / ViT later: pass `feat_dim` explicitly or provide equivalent attr.
    """

    # ------------------------------
    # Construction
    # ------------------------------
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feat_dim: Optional[int] = None,
        partitions: Optional[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        enforce_coverage: bool = True,
        init_heads: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.backbone = backbone

        # --- infer D
        D = None
        prev_fc = getattr(self.backbone, "fc", None)
        if feat_dim is not None:
            D = int(feat_dim)
        elif isinstance(prev_fc, nn.Linear):
            D = int(prev_fc.in_features)
        else:
            raise ValueError("feat_dim not provided and could not infer from backbone.")
        self._D = D

        # --- create registries BEFORE add_dataset is called
        self.heads = nn.ModuleDict()     # must exist before add_dataset
        self.train_scopes: Dict[str, dict] = {}   # must exist before add_dataset

        # --- create initial heads; transplant backbone.fc if it matches ANY requested head
        transplanted = False
        if init_heads:
            transplant_ds = None
            if isinstance(prev_fc, nn.Linear) and prev_fc.in_features == D:
                # pick the first dataset whose class count matches the pretrained fc
                for ds, ncls in init_heads.items():
                    if int(ncls) == int(prev_fc.out_features):
                        transplant_ds = ds
                        break
        
            for ds, ncls in init_heads.items():
                if (not transplanted) and (transplant_ds is not None) and (ds == transplant_ds):
                    # reuse pretrained classifier weights for this dataset
                    self.add_dataset(ds, ncls, head_module=prev_fc)
                    transplanted = True
                else:
                    self.add_dataset(ds, ncls)
        
        # --- replace backbone.fc with Identity AFTER any transplant
        if isinstance(prev_fc, nn.Linear) and prev_fc.in_features == D:
            self.backbone.fc = Identity()

        # --- partition buffers
        self.register_buffer("idx_A", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("idx_B", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("idx_C", torch.empty(0, dtype=torch.long), persistent=True)

        if partitions is not None:
            idx_A, idx_B, idx_C = self._normalize_partitions_input(partitions)
            # NEW: treat empty A/B/C as "no partitions" and skip coverage checks
            if (idx_A.numel() + idx_B.numel() + idx_C.numel()) > 0:
                self._validate_partitions(idx_A, idx_B, idx_C, self._D, enforce_coverage)
                self._set_partitions_buffers(idx_A, idx_B, idx_C)

    # ------------------------------
    # Partitions
    # ------------------------------
    @staticmethod
    def _normalize_partitions_input(partitions):
        if isinstance(partitions, dict):
            idx_A, idx_B, idx_C = partitions["A"], partitions["B"], partitions["C"]
        else:
            idx_A, idx_B, idx_C = partitions
        return idx_A.detach().clone(), idx_B.detach().clone(), idx_C.detach().clone()

    @staticmethod
    def _validate_partitions(idx_A, idx_B, idx_C, D, enforce_coverage=True):
        for t in (idx_A, idx_B, idx_C):
            assert t.dim() == 1 and t.dtype in (torch.int64, torch.int32)
            assert torch.all((t >= 0) & (t < D)), f"indices must be in [0,{D})"
        A, B, C = set(idx_A.tolist()), set(idx_B.tolist()), set(idx_C.tolist())
        assert len(A & B) == 0 and len(A & C) == 0 and len(B & C) == 0, "A,B,C must be disjoint"
        if enforce_coverage:
            assert len(A | B | C) == D, f"A∪B∪C must cover all {D} channels"

    def _set_partitions_buffers(self, idx_A, idx_B, idx_C):
        # Pick a device from backbone params; fall back to any model param; else CPU.
        p = next(self.backbone.parameters(), None)
        if p is None:
            try:
                p = next(self.parameters())
            except StopIteration:
                p = None
        device = p.device if p is not None else torch.device("cpu")
    
        # Replace existing buffers (shape may change).
        for name in ("idx_A", "idx_B", "idx_C"):
            if name in self._buffers:
                del self._buffers[name]
    
        self.register_buffer("idx_A", _as_long(idx_A, device), persistent=True)
        self.register_buffer("idx_B", _as_long(idx_B, device), persistent=True)
        self.register_buffer("idx_C", _as_long(idx_C, device), persistent=True)


    @torch.no_grad()
    def set_partitions(self, idx_A, idx_B, idx_C, enforce_coverage=True):
        self._validate_partitions(idx_A, idx_B, idx_C, self._D, enforce_coverage)
        self._set_partitions_buffers(idx_A, idx_B, idx_C)

    def add_dataset(
        self,
        dataset: str,
        num_classes: int,
        *,
        init: str = "kaiming",
        head_module: Optional[nn.Module] = None,
    ):
        # pick device from backbone params
        dev = next(self.backbone.parameters()).device
    
        if head_module is not None:
            assert isinstance(head_module, nn.Linear), "head_module must be nn.Linear"
            assert head_module.in_features == self._D, f"in_features mismatch: {head_module.in_features} vs {self._D}"
            assert head_module.out_features == int(num_classes), "out_features must match num_classes"
            head = head_module.to(dev)                        # <<< move transplanted head to backbone device
        else:
            head = nn.Linear(self._D, int(num_classes), bias=True).to(dev)  # <<< create on proper device
            if init == "kaiming":
                nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
                nn.init.zeros_(head.bias)
    
        self.heads[dataset] = head
    
        if dataset not in self.train_scopes:
            self.train_scopes[dataset] = {
                "mode": "generic", "allowed_parts": None,
                "param_allowlist": None, "col_mask": None, "hook_handle": None,
            }


    # ------------------------------
    # Feature extraction
    # ------------------------------
    def extract_feats(self, x) -> torch.Tensor:
        """Backbone forward that returns [B, D] pooled features."""
        return self.backbone(x)

    # ------------------------------
    # Forward
    # ------------------------------
    @torch.no_grad()
    def _apply_partition_mask_inplace(self, z: torch.Tensor, allowed_parts: Optional[set]):
        """
        Zero out disallowed partitions INPLACE on z (copy passed in).
        If allowed_parts is None -> keep full z (generic).
        """
        if allowed_parts is None:  # generic mode
            return
        if 'A' not in allowed_parts and self.idx_A.numel() > 0:
            z[:, self.idx_A] = 0.0
        if 'B' not in allowed_parts and self.idx_B.numel() > 0:
            z[:, self.idx_B] = 0.0
        if 'C' not in allowed_parts and self.idx_C.numel() > 0:
            z[:, self.idx_C] = 0.0

    def forward(
        self,
        x: torch.Tensor,
        *,
        dataset: str,
        route: str = "auto"
    ) -> torch.Tensor:
        """
        dataset: which head to use
        route: "auto" | "full" | "masked"
           - "auto": uses the dataset's configured mode/allowed_parts
           - "full": ignore partitions (generic)
           - "masked": zero out disallowed partitions given the dataset's allowed_parts
        """
        if dataset not in self.heads:
            raise KeyError(f"Unknown dataset '{dataset}'. Add it via add_dataset(dataset, num_classes).")

        head = self.heads[dataset]
        z = self.extract_feats(x)  # [B, D]

        # Determine which partitions are allowed for this dataset
        allowed_parts = None
        if route == "full":
            allowed_parts = None
        elif route == "masked" or route == "auto":
            scope = self.train_scopes.get(dataset, {})
            if route == "auto":
                allowed_parts = scope.get("allowed_parts", None)
            else:
                allowed_parts = scope.get("allowed_parts", None)
        # Apply mask by zeroing disallowed parts
        if allowed_parts is not None:
            z = z.clone()
            self._apply_partition_mask_inplace(z, allowed_parts)

        return head(z)

    # ------------------------------
    # Dataset training scopes
    # ------------------------------
    def configure_training_scope(
        self,
        dataset: str,
        *,
        mode: str = "generic",                  # "generic" | "partitioned" | "subset"
        allowed_parts: Optional[Iterable[str]] = None,  # e.g., {"B","C"} for target domain
        param_allowlist: Optional[List[str]] = None,    # regexes for names to train (subset mode)
        head_trainable_columns: Optional[Union[List[int], torch.Tensor]] = None,
        install_grad_hook: bool = True
    ):
        """
        Configure how training should behave for a given dataset:
          - generic: no partition masking, typical finetuning
          - partitioned: use allowed_parts to restrict features; still generic param updates unless you freeze
          - subset: (optionally) restrict *which parameters* are updated via regex allowlist;
                    (optionally) restrict *which input columns of the head* update via a grad mask

        `head_trainable_columns` => only these columns (feature dims) in head.weight will receive gradients.
        """
        if dataset not in self.heads:
            raise KeyError(f"Unknown dataset '{dataset}'.")

        scope = self.train_scopes[dataset]
        scope["mode"] = mode
        scope["allowed_parts"] = set(allowed_parts) if allowed_parts is not None else None
        scope["param_allowlist"] = list(param_allowlist) if param_allowlist else None

        # Remove any old hook
        if scope.get("hook_handle") is not None:
            try:
                scope["hook_handle"].remove()
            except Exception:
                pass
            scope["hook_handle"] = None

        # Column mask for head weight grads
        if head_trainable_columns is not None:
            idx = _as_long(head_trainable_columns, self.heads[dataset].weight.device)
            mask = _make_col_grad_mask(self._D, idx, device=idx.device)
            scope["col_mask"] = mask

            if install_grad_hook:
                def _mask_grad(grad, _mask=mask):
                    return grad * _mask
                handle = self.heads[dataset].weight.register_hook(_mask_grad)
                scope["hook_handle"] = handle
        else:
            scope["col_mask"] = None
            scope["hook_handle"] = None

    def apply_freeze_policy(self, dataset: str):
        """
        Apply a freeze policy according to the dataset scope:
          - generic: do nothing (user can freeze/unfreeze manually if desired)
          - partitioned: also do nothing by default; partitioning is enforced via route/mask
          - subset: freeze everything first, then unfreeze only allowed params (regex by name)
        """
        if dataset not in self.train_scopes:
            raise KeyError(f"No scope configured for dataset '{dataset}'. Call configure_training_scope first.")

        scope = self.train_scopes[dataset]
        mode = scope.get("mode", "generic")
        allowlist = scope.get("param_allowlist", None)

        if mode != "subset":
            return  # nothing to do

        # Freeze all
        for p in self.parameters():
            p.requires_grad_(False)

        # Unfreeze whitelisted parameters by regex on parameter names
        if allowlist:
            for name, p in self.named_parameters():
                if any(re.search(rx, name) for rx in allowlist):
                    p.requires_grad_(True)

    # ------------------------------
    # Quality-of-life helpers
    # ------------------------------
    def head_param_names(self, dataset: str) -> List[str]:
        out = []
        prefix = f"heads.{dataset}."
        for n, _ in self.named_parameters():
            if n.startswith(prefix):
                out.append(n)
        return out

    def freeze_backbone(self, freeze_bn_running_stats: bool = True):
        for p in self.backbone.parameters(): p.requires_grad_(False)
        if freeze_bn_running_stats:
            def _set_bn_eval(m):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
            self.backbone.apply(_set_bn_eval)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad_(True)

    def remove_all_grad_hooks(self):
        for ds, scope in self.train_scopes.items():
            h = scope.get("hook_handle", None)
            if h is not None:
                try: h.remove()
                except Exception: pass
                scope["hook_handle"] = None
            scope["col_mask"] = None