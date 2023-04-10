"""
Performance test for the exact 0-1 loss linear classification algorithm on real-world dataset.

CC BY-SA 4.0, Xi He, Max A. Little, 2023. If you use this code, please cite:
X. He, M.A. Little, 2023, "E01Loss: A Python library for solving the
exact 0-1 loss linear classification problem", doi:10.5281/zenodo.7814259
"""

import e01loss.exact_classify01loss
import numpy as np
import time

data = np.genfromtxt('voicepath_data.csv', delimiter=',')
X = data[:, :2]
y = data[:, 2]
y[y == 0] = -1

start_time = time.time()
optconfig_comb, opt01loss_comb, w_comb, b_comb = e01loss.exact_classify01loss_cell(X, y, display=True, max_margin=True)
end_time = time.time()
print(f"Elapsed time: {end_time-start_time:.2f} seconds")
print(w_comb, b_comb)