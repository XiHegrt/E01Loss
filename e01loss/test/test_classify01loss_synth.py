"""
Performance test for the exact 0-1 loss linear classification algorithm on the synthetic dataset.

CC BY-SA 4.0, Xi He, Max A. Little, 2023. If you use this code, please cite:
X. He, M.A. Little, 2023, "E01Loss: A Python library for solving the
exact 0-1 loss linear classification problem", doi:10.5281/zenodo.7814259
"""

from e01loss.exact_classify01loss import *
import numpy as np
import time

N = 30
D = 3
overlap = 0.15
balance_fac = 0.25

accept = False
while not accept:
    xdata = np.vstack([2.0 * (np.random.rand(D, N) - 0.5), np.ones(N)])
    p0t = 0.8 * (np.random.rand() - 0.5)
    pt = 0.8 * (np.random.rand(D, 1) - 0.5)
    p0t = p0t / np.linalg.norm(pt)
    pt = pt / np.linalg.norm(pt)
    w = np.vstack([p0t, pt])
    ydata = np.sign(w.T @ xdata + overlap * np.random.randn(1, N))
    Npos = np.sum(ydata == 1)
    Nneg = np.sum(ydata == -1)
    accept = (Npos > balance_fac * N) and (Nneg > balance_fac * N)

X_hom = xdata.T
y = ydata.flatten().astype(int)
X = X_hom[:, :-1]

start_time = time.time()
optconfig_comb, opt01loss_comb, w_comb, b_comb = exact_classify01loss_comb(X, y, display=True, max_margin=True)
end_time = time.time()
print(f"Elapsed time: {end_time-start_time:.2f} seconds")
print(optconfig_comb, opt01loss_comb)
print(w_comb, b_comb)

start_time = time.time()
optconfig_purge, opt01loss_purge, w_purge, b_purge = exact_classify01loss_purge(X, y,  display=True, max_margin=True)
end_time = time.time()
print(f"Elapsed time: {end_time-start_time:.2f} seconds")
print(optconfig_purge, opt01loss_purge)
print(w_purge, b_purge)

start_time = time.time()
optconfig_cell, opt01loss_cell, w_cell, b_cell = exact_classify01loss_cell(X, y,  display=False, max_margin=True)
end_time = time.time()
print(f"Elapsed time: {end_time-start_time:.2f} seconds")
print(optconfig_cell, opt01loss_cell)
print(w_cell, b_cell)