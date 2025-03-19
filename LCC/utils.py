import numpy as np
import pandas as pd


def opt_cost_from_counts(counts):
    if len(counts)==1:
        return 0
    if isinstance(counts, list):
        counts = np.array(counts)
    assert counts.ndim == 1
    cost_per_tok = -np.log2((counts/counts.sum()+1e-8))
    total_cost = (cost_per_tok*counts).sum()
    if np.isnan(total_cost):
        breakpoint()
    return total_cost

def opt_cost_from_discrete_seq(t):
    v, counts = np.unique(list(t), return_counts=True)
    return opt_cost_from_counts(counts)
