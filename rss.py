"""Redistribution Sensitivity Score (RSS).

For one prompt and one transformation, RSS keeps the SAE features that change a lot,
in most variants, and in the same direction. MeanRSS is their mean change.
"""
import numpy as np


def rss(delta, rho=0.8, kappa=0.8, tau=1e-5, M=400, K=20):
    """Select the responsive support of one prompt under one transformation.

    delta: array of shape (n_variants, n_features), delta[v, f] = a_f(x_v) - a_f(x_0).
    Returns (support, direction, mean_rss): feature ids sorted by change size,
    their dominant direction (+1 / -1), and MeanRSS (nan when nothing is selected).
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.shape[0]
    h = np.abs(delta).mean(axis=0)                     # mean absolute change
    up = (delta > tau).sum(axis=0)                     # variants where f goes up
    down = (delta < -tau).sum(axis=0)                  # variants where f goes down
    moved = up + down
    r = moved / n                                      # recurrence
    c = np.divide(np.maximum(up, down), moved, out=np.zeros(h.shape), where=moved > 0)  # consistency

    pool = np.argsort(-h, kind="stable")[:M]           # M largest changes
    ok = pool[(r[pool] >= rho) & (c[pool] >= kappa)]   # recurrent and one-directional
    ok = ok[np.lexsort((ok, -c[ok], -h[ok]))]          # largest h first; ties: consistency, id
    support = ok[:K]

    direction = np.sign(up[support] - down[support]).astype(int)
    mean_rss = float(h[support].mean()) if len(support) else float("nan")
    return support, direction, mean_rss
