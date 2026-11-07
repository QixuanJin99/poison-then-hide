# clip_backbone.py
import torch
import torch.nn as nn


class TimmBackboneWrapper(nn.Module):
    """
    Wrap a timm classifier to expose forward(x)->[B, D] pooled features (float32),
    and a .feat_dim attribute. Works for ViT/ConvNeXt/etc. in timm.
    """
    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.m = timm_model
        # timm models expose num_features for the pooled feature dimension
        self.feat_dim = int(getattr(self.m, "num_features"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.m(x)                # with num_classes=0 => [B, D]
        return feats.float()


# Works for openai CLIP based models 
class ClipVisualWrapper(nn.Module):
    """
    Thin wrapper around an OpenCLIP / CLIP model that:
      - exposes a forward(x) -> [B, D] pooled image embedding (float32)
      - stores self.feat_dim for downstream models
      - has no .fc (so MultiDomainPartitionedModel won't try to transplant)
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        # Try common ways to read the image embedding dimension
        D = None
        # open_clip: text_projection is [D_text, D], use the 2nd dim
        proj = getattr(self.clip, "text_projection", None)
        if proj is not None and isinstance(proj, torch.Tensor) and proj.ndim == 2:
            D = int(proj.shape[1])
        # fallback: some models expose visual.output_dim
        if D is None:
            visual = getattr(self.clip, "visual", None)
            if hasattr(visual, "output_dim"):
                D = int(visual.output_dim)
        # hard fail if still unknown
        if D is None:
            raise ValueError("Could not infer CLIP image embedding dimension; "
                             "set it manually or extend this wrapper.")
        self.feat_dim = D  # used below

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Keep CLIP’s usual behavior (may be fp16 under AMP)
        z = self.clip.encode_image(x)  # [B, D], usually L2-normalized
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure downstream Linear heads (float32) are happy regardless of AMP/fp16
        z = self.encode_image(x)
        return z.float()  