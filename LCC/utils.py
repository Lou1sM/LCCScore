import numpy as np
import pandas as pd
from line_profiler import LineProfiler


def profile_aggregate(func):
    """
    Decorator that accumulates timing stats across multiple function calls
    and prints results only at the end of the program
    """
    # Create a single profiler instance that persists across calls
    profiler = LineProfiler()
    wrapped = profiler(func)

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    # Store the profiler so we can access it later
    wrapper.profiler = profiler
    return wrapper

def profile_lines(func):
    """
    Decorator to profile specific lines within a function
    """
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler_wrapper = profiler(func)
        result = profiler_wrapper(*args, **kwargs)

        print("\n=== Line-by-line profiling ===")
        profiler.print_stats()

        return result
    return wrapper

def print_stats(func):
    """Call this after all executions to see accumulated stats"""
    if hasattr(func, 'profiler'):
        func.profiler.print_stats()
    else:
        print("No profiler found on function")

def make_alls_df(df):
    inner_alls_dfs = [df.loc[k,'all'] for k in df.index.levels[0]]
    return pd.concat(inner_alls_dfs,keys=df.index.levels[0],axis=1).T

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

