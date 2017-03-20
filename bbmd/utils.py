import numpy as np
from scipy import stats


def get1Dkernel(arr, lowp=0.01, highp=99.99, steps=1024j):
        lower = np.percentile(arr, lowp)
        upper = np.percentile(arr, highp)
        density = stats.gaussian_kde(arr)
        x = np.r_[lower:upper:steps]
        y = density.pdf(x)
        return {
            'x': x,
            'y': y,
        }


def get_summary_stats(arr):
    return {
        'mean': np.nanmean(arr),
        'std':  np.nanstd(arr),
        'p5':   np.nanpercentile(arr, 5),
        'p25':  np.nanpercentile(arr, 25),
        'p50':  np.nanpercentile(arr, 50),
        'p75':  np.nanpercentile(arr, 75),
        'p95':  np.nanpercentile(arr, 95),
    }
