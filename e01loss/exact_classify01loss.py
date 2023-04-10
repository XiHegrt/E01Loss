"""
Exact 0-1 loss linear classification algorithms.

CC BY-SA 4.0, Xi He, Max A. Little, 2023. If you use this code, please cite:
X. He, M.A. Little, 2023, "E01Loss: A Python library for solving the
exact 0-1 loss linear classification problem", doi:10.5281/zenodo.7814259
"""


from operator import itemgetter
from e01loss.auxfuncs import *
from e01loss.classifylinear import classifylinear
import itertools
import numpy as np

def exact_classify01loss_comb(X, y, display=False, svm_C= 1, svm_eps=1e-3, max_margin = True):
    """
    Combinatorial exact linear 0-1 loss classification.
    Inputs:  X  - NxD, matrix of data
             y  - Nx1, label vector (1: positive label, -1: negative label)
             display  - bool, default = False, display the number of configurations before and after each step,
                        in format num1 (num2 -> num3), where
                        num1: input data item number
                        num2: number of configurations before filtering
                        num3: number of configurations after filtering
             svm_C  - int, default = 2e6, regularization parameter for SVM
             svm_eps  - int, default = 1e-3, tolerance level for the convergence of the SVM al
    Outputs: optconfig[0] - list, optimal configuration
             opt01loss  - int, optimal 0-1 loss
    """

    # Transform data from inhomogeneous coordinates to homogeneous coordinates
    N, D = X.shape
    one_col = np.ones((N, 1))
    X = np.append(X, one_col, axis=1)

    # Obtain approximate (SVM) upper bound on 0-1 loss
    model = classifylinear(C=svm_C)
    model.fit(X, y)
    ub = model.loss01(X, y)

    if display:
        print(f"SVM upper bound 0-1 loss: {ub}")

    # Transform numpy array to list
    X = X.tolist()
    y = list(y)

    # Initial configuration list
    empty = ([], 0)
    configs = [empty]

    # Process the data incrementally
    for n in range(0, N):
        Nconfigs = len(configs)

        # Update configuration list and associated 0-1 loss, append +1 or -1 label to every current configuration
        config_update = list(map(lambda f: list(map(lambda config: f(n, y, config), configs)), update_funcs))
        configs = flatten(config_update)

        # Filter out all infeasible configurations: those which are not linearly separable or lower than the upper bound ub
        configs = list(filter(lambda config: filt_func(X, n, ub, config), configs))
        if display:
            print(f"Processing sample: {n} ({Nconfigs} -> {len(configs)})")

    # Select the optimal configuration from configs
    opt01loss = math.inf
    optconfig = None
    for n in range(0, len(configs)):
        if (configs[n][1] < opt01loss):
            opt01loss = configs[n][1]
            optconfig = configs[n]

    # Output maximum margin decision boundary or arbitrary boundary obtained by linear programming
    if N <= D:
        return None
    else:
        if max_margin:
            X = np.array(X)
            X = X[:, :-1]
            y = optconfig[0]
            y = np.array(y)
            model = classifylinear()
            w, b = model.fit(X, y, max_margin=True)
        else:
            X = np.array(X)
            X = X[:, :-1]
            y = optconfig[0]
            y = np.array(y)
            model = classifylinear()
            w, b = model.fit(X, y, max_margin=False)

    if display:
        print(f"Exact 0-1 loss (exact_classify01loss_comb): {opt01loss}")

    return optconfig[0], opt01loss, w, b


def exact_classify01loss_purge(X, y, display=False, svm_C=1, svm_eps=1e-3, max_margin = True):
    """
    Combinatorial exact linear 0-1 loss classification, using finite-length dominance-based thinning.
    Inputs:  X  - NxD, matrix of data
             y  - Nx1, label vector (1: positive label, -1: negative label)
             display  - bool, default = False, display the number of configurations before and after each step,
                        in format num1 (num2 -> num3), where
                        num1: input data item number
                        num2: number of configurations before filtering and dominance thinning
                        num3: number of configurations after filtering and dominance thinning
             svm_C  - int, default = 1, regularization parameter for SVM
             svm_eps  - int, default = 1e-3, tolerance level for the convergence of the SVM algorithm
    Outputs: (list, int) optimal configuration, optimal 0-1 loss
    """

    # Transform data from inhomogeneous coordinates to homogeneous coordinates
    N, D = X.shape
    one_col = np.ones((N, 1))
    X = np.append(X, one_col, axis=1)

    # Obtain approximate (SVM) upper bound on 0-1 loss
    model = classifylinear(C=svm_C)
    model.fit(X, y)
    ub = model.loss01(X, y)

    if display:
        print(f"SVM upper bound 0-1 loss: {ub}")

    # Transform numpy array to list
    X = X.tolist()
    y = list(y)

    # Initial configuration list
    empty = ([], 0)
    configs = [empty]

    # Process each input data item incrementally
    for n in range(0, N):
        Nconfigs = len(configs)

        # Update configuration list and associated 0-1 loss , append +1 or -1 label to the current configurations
        config_update = list(map(lambda f: list(map(lambda config: f(n, y, config), configs)), update_funcs))
        configs = flatten(config_update)

        # Filter out all infeasible configurations: those which are not linearly separable or lower than the upper bound ub
        configs = list(filter(lambda config: filt_func(X, n, ub, config), configs))

        # Dominance-based thinning
        configs = sorted(configs, key=itemgetter(1))  # Sort the list to speed up thinning process

        # Remove all configurations that are dominated by other configurations using finite dominance principle
        configs = config_thin_sort(configs, n, N)

        if display:
            print(f"Processing sample: {n} ({Nconfigs} -> {len(configs)})")

    # at this point, all non-optimal configurations have been removed, only optimal configuration left in configs
    optconfig = configs[0]
    opt01loss = configs[0][1]

    # Output maximum margin decision boundary or arbitrary boundary obtained by linear programming
    if N <= D:
        return None
    else:
        if max_margin:
            X = np.array(X)
            X = X[:, :-1]
            y = optconfig[0]
            y = np.array(y)
            model = classifylinear()
            w, b = model.fit(X, y, max_margin=True)
        else:
            X = np.array(X)
            X = X[:, :-1]
            y = optconfig[0]
            y = np.array(y)
            model = classifylinear()
            w, b = model.fit(X, y, max_margin=False)

    if display:
        print(f"Exact 0-1 loss (exact_classify01loss_thin): {opt01loss}")

    return optconfig[0], opt01loss, w, b


def exact_classify01loss_cell(X, y, display = False, max_margin = True):
    """
    cell enumeration algorithm for 0-1 loss classification problem.
    Inputs:  X  - NxD, matrix of data (inhomogeneous coordinates, i.e. without extra dimension for fixed value 1)
             y  - Nx1, label vector (1: positive label, -1: negative label)
             display  - bool, default = False, print the optimal 0-1 loss value
    Outputs: optconfig[0] - list, optimal configuration
             opt01loss  - int, optimal 0-1 loss value
    """

    ## Input X with non-homogeneous coordinates input X
    N, D = X.shape
    # Transform points D to the N dual hyperplanes Phi(D), the last coordinates of each point is the intercept of the corresponding hyperplane

    PhiD_A = X[:, :-1]
    PhiD_b = X[:, -1]

    # Calculate the intersection vertices S_V of the Nd hyperplane arrangements, and then compute the sign vector of each
    list_N = [i for i in range(0, N)]
    combinations = list(itertools.combinations(list_N, D))
    SV_comb = np.array(combinations)
    Nd = SV_comb.shape[0]
    SV = np.zeros((D, Nd))  # (D*Nd) intersection vertex coordinates
    C = np.hstack((PhiD_A, -np.ones((N, 1))))
    for i in range(Nd):
        SV[:, i] = np.linalg.solve(C[SV_comb[i], :], PhiD_b[SV_comb[i]])

    # Compute the sign vector of each intersection vertex in S_V. To do this, first calculate the signed distance between each Nd intersection vertex to each of the N dual hyperplanes
    SV_signdist = np.dot(C, SV) - np.tile(PhiD_b, (Nd, 1)).T
    numtol = 1e-10
    SV_signdist[np.abs(SV_signdist) < numtol] = 0  # Correct for numerical inaccuracy
    SV_possign = np.sign(SV_signdist)

    # Calculate the implied 0-1 loss of these sign vectors
    YNd = np.tile(y, (Nd, 1)).T
    loss01_SV_pos = np.sum(SV_possign != YNd, axis=0)
    loss01_SV_neg = np.sum((-SV_possign) != YNd, axis=0)

    # Select the sign vectors corresponding to the optimal 0-1 loss, for both positive and negative vectors
    loss01_minpos, min_pos_index = np.min(loss01_SV_pos), np.argmin(loss01_SV_pos)
    loss01_minneg, min_neg_index = np.min(loss01_SV_neg), np.argmin(loss01_SV_neg)

    # Choose the best of either positive or negative sign vectors
    if loss01_minpos < loss01_minneg:
        SV_opt_index = min_pos_index
        SV_optclass = 1  # Optimal vertex belongs to positive sign vector
    else:
        SV_opt_index = min_neg_index
        SV_optclass = -1  # Optimal vertex belongs to negative sign vector

    # Resolve zero sign vectors of intersection vertices. There are 2^D ways to do this, each corresponding to an adjacent cell, S_C, for each intersection vertex in S_V

    Nc = 2 ** D
    SC_enum = np.array(list(itertools.product([-1, 1], repeat=D)))

    # Test every one of the 2^D adjacent cells in S_C, select the best cell, i.e. the cell whose sign vectors imply the best 0-1 loss
    loss01_opt = math.inf
    assign_opt = np.zeros(N)
    if SV_optclass == 1:
        SC_assign = SV_possign[:, SV_opt_index].copy()
    else:
        SC_assign = -SV_possign[:, SV_opt_index].copy()

    for i in range(Nc):
        ii = SV_comb[SV_opt_index]
        SC_assign[ii] = SC_enum[i]

        loss01_adjcell = sum(SC_assign != y)

    # Find the optimal configuration in S_C
        if (loss01_adjcell < loss01_opt):
            assign_opt = SC_assign
            assign_opt = assign_opt
            loss01_opt = loss01_adjcell

    if display:
        print(f"Exact 0-1 loss: {loss01_opt}")
    assign_opt = assign_opt.astype(int)
    assign_opt = list(assign_opt)

    # Output maximum margin decision boundary or arbitrary boundary obtained by linear programming
    if N <= D:
        return None
    else:
        if max_margin:
            model = classifylinear()
            y = assign_opt
            y = np.array(y)
            w, b = model.fit(X, y, max_margin=True)
        else:
            model = classifylinear()
            y = assign_opt
            y = np.array(y)
            w, b = model.fit(X, y, max_margin=False)

    return assign_opt, loss01_opt, w, b