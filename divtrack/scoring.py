"""Running the networks on a sequence.

The batching mirrors the original implementation (which matters for bit-identical results):
    judge_scores      per frame pair, candidates in solver order, chunks of ``score_chunk``,
                      float32 patches; sigmoid computed in float32 numpy
    encode_cells      cells in table order, batches of ``encode_batch``, float16-rounded patches
    state_scores      chunks of ``state_batch``
    judge_logits      candidate table order (daughters sorted by label), chunks of ``batch``,
                      float16-rounded patches; raw logits (used to fit the fusion weight)
"""
import numpy as np
import torch
import torch.nn.functional as F


def judge_scores(net, p32, cells, cands, device, chunk=512, log=None):
    """Judge probabilities of every division candidate: list over frame pairs (float64 arrays)."""
    net.eval()
    out = []
    for fi, p in enumerate(cands.pairs):
        if p is None or len(p["tri_i"]) == 0:
            out.append(np.zeros(0, np.float64))
            continue
        a0, b0 = cells.start[cands.frames[fi - 1]], cells.start[cands.frames[fi]]
        P = np.stack([p32[a0 + p["tri_i"]], p32[b0 + p["tri_j"]], p32[b0 + p["tri_k"]]], axis=1)
        PT = torch.from_numpy(P).float()
        with torch.no_grad():
            lg = np.concatenate([np.atleast_1d(net(PT[s:s + chunk].to(device)).cpu().numpy())
                                 for s in range(0, len(PT), chunk)])
        sc = 1.0 / (1.0 + np.exp(-lg))
        out.append(sc.astype(np.float64))
        if log is not None and fi % 50 == 0:
            log("  scored frame pair %d/%d" % (fi, len(cands.pairs) - 1))
    return out


def _features(net, P16, device):
    x = torch.from_numpy(np.asarray(P16).astype(np.float32)).to(device)[:, None]
    with torch.no_grad():
        h = net.enc(x)
        h = F.adaptive_avg_pool3d(h, (2, 2, 2)).flatten(1)
        return net.proj(h).cpu().numpy()


def encode_cells(net, p32, prev16, next16, device, batch=512):
    """Frozen-encoder features of every cell at (t-1, t, t+1): (N, 3, feature_dim) float32."""
    net.eval()
    N = len(p32)
    out = np.zeros((N, 3, net.feature_dim), np.float32)
    for a in range(0, N, batch):
        b = slice(a, min(a + batch, N))
        out[b, 1] = _features(net, np.asarray(p32[b]).astype(np.float16), device)
        out[b, 0] = _features(net, prev16[b], device)
        out[b, 2] = _features(net, next16[b], device)
    return out


def state_scores(head, feats, valid, device, batch=2048):
    """u for every cell (float32)."""
    head.eval()
    Fx = torch.from_numpy(np.asarray(feats, np.float32)).to(device)
    Vx = torch.from_numpy(np.asarray(valid, bool)).to(device)
    out = []
    with torch.no_grad():
        for a in range(0, len(Fx), batch):
            out.append(head(Fx[a:a + batch], Vx[a:a + batch]).cpu().numpy())
    u = np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
    assert np.isfinite(u).all()
    return u


def judge_logits(net, p16_table, tri_rows, device, batch=4096):
    """Raw judge logits of the candidate table (rows = mother, daughter, daughter), float64."""
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(tri_rows), batch):
            ix = np.arange(a, min(a + batch, len(tri_rows)))
            P = torch.from_numpy(np.asarray(p16_table[tri_rows[ix].reshape(-1)]).astype(np.float32)).to(device)
            P = P.reshape(len(ix), 3, *p16_table.shape[1:])
            out.append(net(P).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
