"""Models B-E: MobileNetV3-Small, MobileViT-XS, MobileViT-XS + late fusion, MobileViT-XS + DIP tokens."""
import timm
import torch
import torch.nn as nn

from src.dip.features import FEATURE_GROUPS

GROUP_SIZES = [len(v) for v in FEATURE_GROUPS.values()]
N_FEAT = sum(GROUP_SIZES)


class ImageOnly(nn.Module):
    def __init__(self, arch: str, pretrained: bool, n_classes: int = 3):
        super().__init__()
        self.net = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)

    def forward(self, x, f=None):
        return self.net(x)


class LateFusion(nn.Module):
    """Image embedding and the 16 DIP features joined just before the classifier."""

    def __init__(self, arch: str, pretrained: bool, n_classes: int = 3, feat_dim: int = 64, drop: float = 0.2):
        super().__init__()
        self.net = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.fmlp = nn.Sequential(nn.LayerNorm(N_FEAT), nn.Linear(N_FEAT, feat_dim), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.net.num_features + feat_dim, n_classes))

    def forward(self, x, f):
        return self.head(torch.cat([self.net(x), self.fmlp(f)], dim=1))


class HintTokens(nn.Module):
    """Wraps a MobileViT transformer: one token per DIP feature group joins the patch tokens in every layer."""

    def __init__(self, seq: nn.Module, dim: int):
        super().__init__()
        self.seq = seq
        self.proj = nn.ModuleList([nn.Linear(g, dim) for g in GROUP_SIZES])
        self.group_emb = nn.Parameter(torch.zeros(1, len(GROUP_SIZES), dim))
        nn.init.trunc_normal_(self.group_emb, std=0.02)
        self.bounds = torch.tensor([0, *GROUP_SIZES]).cumsum(0).tolist()
        self.feats = None

    def forward(self, x):
        f = self.feats
        tok = torch.stack([p(f[:, a:b]) for p, a, b in zip(self.proj, self.bounds[:-1], self.bounds[1:])], 1)
        tok = (tok + self.group_emb).to(x.dtype).repeat_interleave(x.shape[0] // f.shape[0], dim=0)
        k = tok.shape[1]
        return self.seq(torch.cat([tok, x], dim=1))[:, k:]


class DipTokens(nn.Module):
    def __init__(self, arch: str, pretrained: bool, n_classes: int = 3):
        super().__init__()
        self.net = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)
        self.hints = nn.ModuleList()
        for m in self.net.modules():
            if type(m).__name__ == "MobileVitBlock":
                m.transformer = HintTokens(m.transformer, m.transformer[0].attn.qkv.in_features)
                self.hints.append(m.transformer)
        if not self.hints:
            raise ValueError(f"{arch} has no MobileViT blocks")

    def forward(self, x, f):
        for h in self.hints:
            h.feats = f
        out = self.net(x)
        for h in self.hints:
            h.feats = None
        return out


def build_model(kind: str, arch: str, pretrained: bool) -> nn.Module:
    return {"image": ImageOnly, "concat": LateFusion, "tokens": DipTokens}[kind](arch, pretrained)


def uses_features(kind: str) -> bool:
    return kind in ("concat", "tokens")
