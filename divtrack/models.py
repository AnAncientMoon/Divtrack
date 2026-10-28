"""The two learned parts of the division score F_ijk = z_ijk + w_u * u_i.

TripletNet (z): a shared 3D CNN encodes the patches of mother i (frame t) and of the candidate
daughters j, k (frame t+1) into 96-d features f; the head sees [f_i, (f_j + f_k)/2, |f_j - f_k|]
(symmetric in the daughters) and returns one logit. No geometric quantity is given to the network.

MotherStateHead (u): the frozen encoder of the triplet network embeds the image at the position of
cell i in frames t-1, t and t+1; a single attention layer with frame t as query pools the three
tokens and a small MLP returns one logit ("is i about to divide").

Module and attribute names match the released checkpoints, and modules are created in the
order of the original implementation so that a given seed gives the same initial weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
    def __init__(self, feature_dim=96):
        super().__init__()

        def block(c_in, c_out, stride):
            return nn.Sequential(nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                                 nn.InstanceNorm3d(c_out, affine=True), nn.GELU())

        self.feature_dim = feature_dim
        self.enc = nn.Sequential(block(1, 16, (1, 1, 1)), block(16, 32, (1, 2, 2)),
                                 block(32, 64, (2, 2, 2)), block(64, 64, (1, 2, 2)))
        self.proj = nn.Linear(64 * 8, feature_dim)
        self.head = nn.Sequential(nn.Linear(3 * feature_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def features(self, x):
        """x: (N, 1, Z, Y, X) -> (N, feature_dim)."""
        h = self.enc(x)
        h = F.adaptive_avg_pool3d(h, (2, 2, 2)).flatten(1)
        return self.proj(h)

    def forward(self, P):
        """P: (B, 3, Z, Y, X) patches of (mother, daughter, daughter) -> (B,) logits."""
        B = P.shape[0]
        h = self.enc(P.reshape(B * 3, 1, *P.shape[2:]))
        h = F.adaptive_avg_pool3d(h, (2, 2, 2)).flatten(1)
        f = self.proj(h).reshape(B, 3, -1)
        z = [f[:, 0], (f[:, 1] + f[:, 2]) / 2, (f[:, 1] - f[:, 2]).abs()]
        return self.head(torch.cat(z, 1)).squeeze(-1)


class MotherStateHead(nn.Module):
    def __init__(self, d=96, hidden=32, heads=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(d)
        self.shared_mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.time_positions = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.state_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, f, valid):
        """f: (B, 3, d) features at (t-1, t, t+1); valid: (B, 3) bool. Returns (B,) logits."""
        clean = f.masked_fill(~valid[..., None], 0)
        tokens = self.shared_mlp(self.input_norm(clean)) + self.time_positions[None]
        cur = tokens[:, 1:2]
        pooled, _ = self.attention(cur, tokens, tokens, key_padding_mask=~valid, need_weights=False)
        return self.head(self.state_norm(cur[:, 0] + pooled[:, 0])).squeeze(-1)


def _state_dict(ckpt):
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_triplet_net(path, device="cpu"):
    """Returns (network in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = _state_dict(ckpt)
    if sd["head.0.weight"].shape[1] != 3 * sd["proj.weight"].shape[0]:
        raise ValueError("%s is not an image-only triplet network checkpoint" % path)
    net = TripletNet(feature_dim=int(sd["proj.weight"].shape[0]))
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), ckpt


def load_state_head(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = _state_dict(ckpt)
    heads = int(ckpt.get("heads", 4)) if isinstance(ckpt, dict) else 4      # released heads use 4
    net = MotherStateHead(d=int(sd["input_norm.weight"].shape[0]), hidden=int(sd["time_positions"].shape[1]), heads=heads)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), ckpt
