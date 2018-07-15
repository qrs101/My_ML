import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x, y, use_normal_equation=True, eta=0.1, deteailed=False):
        N = x.shape[0]
        X = np.hstack((np.ones([N, 1]), x))
        if use_normal_equation:
            inv_matrix = np.linalg.inv(np.dot(X.T, X))
            self.w = np.dot(np.dot(inv_matrix, X.T), np.reshape(y, [-1, 1]))
            self.w = self.w.flatten()
            self.b = self.w[0]
            self.w = self.w[range(1, len(self.w))]
        else:
            pass

    def predict(self, x, detailed=False):
        return np.sum(self.w * x, axis=-1) + self.b