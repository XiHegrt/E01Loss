"""
Algorithms for fitting hard/soft-SVM and obtaining maximal margin/arbitrary decision boundary.

CC BY-SA 4.0, Xi He, Max A. Little, 2023. If you use this code, please cite:
X. He, M.A. Little, 2023, "E01Loss: A Python library for solving the
exact 0-1 loss linear classification problem", doi:10.5281/zenodo.7814259
"""

import cvxopt
import numpy as np


class classifylinear(object):
  """
      This class use the CVXOPT library to slove the quadratic programming problem in SVM and
      linear programming problem that used to find an arbitrary decision boundary and test if
      a data set is linear separable.

      Parameters
      ----------
      C  - float, default = None, regularization parameter. When C = None, the fit function
          invoke a Hard-SVM model, when C = float, the fit function invoke a Soft-SVM model
      Max_margin  - bool, default = True, use CVXOPT library to solve a quadratic programming
                   probelm and return the weight parameter w and intercept parameter if True,
                   solve a linear programming problem to obtain an arbitrary decision boundary,
                   if False.
     """

  def __init__(self, C=None):


    self.C = C
    if self.C is not None: self.C = float(self.C)

  def fit(self, X, y, max_margin=True):
    """
    Fit the model according to the given training data. Solving a quadratic programming problem
    in SVM or obtaining linear separable decison hyperplane by solving a linear program.

    Parameters
    ----------
    Inputs:  X  - NxD, matrix of data (inhomogeneous coordinates,
                  i.e. without extra dimension for fixed value 1).
             y  - Nx1, label vector (1: positive label, -1: negative label).
             Max_margin  - bool, default = True, use CVXOPT library to solve a quadratic programming
                   probelm and return the weight parameter w and intercept parameter if True,
                   solve a linear programming problem to obtain an arbitrary decision boundary,
                   if False.
    Outputs: w  - Dx1, weight vector of the decision boundary.
             b  - float, intercept of the decision boundary
    """
    if max_margin:
        n_samples, n_features = X.shape
        y = y.astype(float)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j].T)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Turn off the output of the optimization process
        cvxopt.solvers.options['show_progress'] = False

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        return self.w, self.b
    else:
        # Find an arbitray decision hyperplane by solving a linear program. Given training set X and label vector y.
        n_samples = X.shape[0]
        one_col = np.ones((N, 1))
        X = np.append(X, one_col, axis=1)
        n_features = X.shape[1]

        c = cvxopt.matrix(np.zeros(n_features))
        diagy = cvxopt.matrix(np.diag(y))
        A = cvxopt.matrix(-np.matmul(diagy, X))
        b = cvxopt.matrix(-np.ones(n_samples))
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.lp(c, A, b)
        w = np.array(sol['x'][:-1])
        w = np.squeeze(w)
        b = np.array(sol['x'][-1])
        return w, b


  def project(self, X):
    if self.w is not None:
      return np.dot(X, self.w) + self.b
    else:
      y_predict = np.zeros(len(X))
      for i in range(len(X)):
        s = 0
        for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
          s += a * sv_y * np.dot(X[i], sv.T)
        y_predict[i] = s
      return y_predict + self.b

  def predict(self, X):
    return np.sign(self.project(X)).astype(int)

  def loss01(self, X, y):
    y_predict = self.predict(X)
    loss = 0
    for i in range(len(y_predict)):
      if y_predict[i] != y[i]:
        loss += 1
    return loss


  @staticmethod
  def classify_feasible(X, y):
      """
      Check if a set of data X is linearly separable with respect to labels y.
      Inputs: X  - configuration list, a set of data items (homogeneous coordinates,
                       i.e. with extra dimension for fixed value 1)
              y  - list of labels, associated class label for each data item
      Outputs: bool, True if data X, y is linearly separable.
      """

      X = np.array(X)
      y = np.array(y)
      D = len(X[0, :])
      N = len(X)
      if N <= D:
          return True
      else:
          c = cvxopt.matrix(np.zeros(D))
          diagy = cvxopt.matrix(np.diag(y))
          A = cvxopt.matrix(-np.matmul(diagy, X))
          b = cvxopt.matrix(-np.ones(N))
          cvxopt.solvers.options['show_progress'] = False
          sol = cvxopt.solvers.lp(c, A, b)
          return sol['status'] == 'optimal'

