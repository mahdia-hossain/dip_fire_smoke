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


class DipTokensMaps(DipTokens):
    """MobileViT (with or without DIP tokens) plus an auxiliary head predicting coarse fire and smoke maps."""

    def __init__(self, arch: str, pretrained: bool, n_classes: int = 3, map_stage: int = 3, tokens: bool = True):
        if tokens:
            super().__init__(arch, pretrained, n_classes)
        else:
            nn.Module.__init__(self)
            self.net = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)
            self.hints = nn.ModuleList()
        self.map_stage = map_stage
        with torch.no_grad():
            was = self.net.training
            self.net.eval()
            x = self.net.stem(torch.zeros(1, 3, 64, 64))
            for h in self.hints:
                h.feats = torch.zeros(1, N_FEAT)
            for stage in self.net.stages[:map_stage + 1]:
                x = stage(x)
            for h in self.hints:
                h.feats = None
            self.net.train(was)
        ch = x.shape[1]
        self.map_head = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.SiLU(),
                                      nn.Conv2d(ch, 2, 1))

    def forward(self, x, f=None):
        return self.forward_with_maps(x, f)[0]

    def forward_with_maps(self, x, f):
        for h in self.hints:
            h.feats = f
        x = self.net.stem(x)
        mid = None
        for i, stage in enumerate(self.net.stages):
            x = stage(x)
            if i == self.map_stage:
                mid = x
        logits = self.net.forward_head(self.net.final_conv(x))
        for h in self.hints:
            h.feats = None
        return logits, self.map_head(mid)


def build_model(kind: str, arch: str, pretrained: bool) -> nn.Module:
    if kind == "image_maps":
        return DipTokensMaps(arch, pretrained, tokens=False)
    return {"image": ImageOnly, "concat": LateFusion, "tokens": DipTokens,
            "tokens_maps": DipTokensMaps}[kind](arch, pretrained)


def uses_features(kind: str) -> bool:
    return kind in ("concat", "tokens", "tokens_maps")


def has_maps(kind: str) -> bool:
    return kind in ("tokens_maps", "image_maps")
