"""
Auxiliary functions for exact 0-1 loss linear classification algorithms.

CC BY-SA 4.0, Xi He, Max A. Little, 2023. If you use this code, please cite:
X. He, M.A. Little, 2023, "E01Loss: A Python library for solving the
exact 0-1 loss linear classification problem", doi:10.5281/zenodo.7814259
"""


import math
from e01loss.classifylinear import *

upper_bound = lambda config, ub: config[1] <= ub  # Check if the 0-1 loss of a configuration is smaller than the (heuristic) upper bound
loss01 = lambda y1,y2: (y1 != y2)
subs01loss_pos = lambda n, y, config: (config[0]+[1], config[1]+loss01(y[n],1))
subs01loss_neg = lambda n, y, config: (config[0]+[-1], config[1]+loss01(y[n],-1))
update_funcs = [subs01loss_pos, subs01loss_neg]
filt_func = lambda X, n, ub, config: classifylinear.classify_feasible(X[0:n+1], config[0]) & upper_bound(config, ub)

def config_binary_search(x,targ):
    """
    Binary search algorithm, find the first entry in a ordered list x that has bigger 0-1 loss value than
    targ by using binary search.
    Inputs: x  - list of configurations, this is usually our configs list, which contains a list of candidate
                 configurations.
            targ  - int, the value we are trying to search
    Outputs: L  - int, the first entry index such that the entry has value greater than targ
    """
    Nx = len(x)
    L = 0
    R = Nx - 1
    if x[R][1] < targ:
        L = Nx -1
    else:
        while L < R:
             m = math.floor((L + R)/2)
             if x[m][1] < targ:
                 L = (m + 1)
             else:
                 R = m
    return L


def config_thin_sort(x, n, N):
    """
    Remove ('thin') all configurations in the configs list x that are sorted by their 0-1 loss,
    that are dominated by other configurations by using the finite-length dominance principle.
    Inputs:     x  - list of configurations, configs list, which contains a list of candidate
                     configurations.
                n  - int, the nth iteration in exact_classify01loss_comb/purg algorithm
                N  - int, the size of the classification data
    Outputs:    z  - list of configurations, the config list after purging.
   """

    Nx = len(x)
    if Nx == 0:
        z = []
    elif Nx == 1:
        z = x
    else:
        min_loss = x[0][1]
        if x[-1][1] < min_loss + N - n - 1:
            index = len(x)
        else:
            index = config_binary_search(x, min_loss + N - n - 1 )
        if index > 0:
            z = x[0:index]
        else:
            z = [1]
            z[0] = x[0]
    return z

# Transform a list of lists to a list
def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]

