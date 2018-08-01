import numpy as np
from model.error import Error

class LinearRegression:
    def __init__(self, eta=0.1, epoch=500, batch_size=1):
        self.w = None
        self.b = None
        self.eta = eta
        self.epoch = epoch
        self.batch_size = batch_size

    def fit(self, x, y, use_normal_equation=True, regularization=None, lam=0.1, detailed=False):
        if use_normal_equation:
            X = np.hstack((np.ones([x.shape[0], 1]), x))
            if regularization is None:
                inv_matrix = np.dot(X.T, X)
            elif regularization == 2:
                inv_matrix = np.dot(X.T, X) + lam * np.eye(X.shape[1])
            else:
                raise Error("invalid regularization!")
            if np.linalg.det(inv_matrix) == 0:
                raise Error("The matrix is singular, can't do inverse!")
            inv_matrix = np.linalg.inv(inv_matrix)
            self.w = np.dot(np.dot(inv_matrix, X.T), np.reshape(y, [-1, 1]))
            self.w = self.w.flatten()
            self.b = self.w[0]
            self.w = self.w[range(1, len(self.w))]
        else:
            if self.batch_size is None:
                self.batch_size = x.shape[0]

            self.w = np.ones(x.shape[1])
            self.b = np.ones(1)
            num_of_batch = x.shape[0] // self.batch_size
            if x.shape[0] % self.batch_size != 0:
                num_of_batch += 1

            for i in range(self.epoch):
                for j in range(num_of_batch):
                    start = j * self.batch_size
                    end = min((j + 1) * self.batch_size, x.shape[0])
                    error = self.predict(x[start: end, :]) - y[start: end]
                    error = error.reshape([-1, 1])
                    if regularization is None:
                        delta_w = np.sum(error * x[start: end, :], axis=0)
                        delta_b = np.sum(error)
                    elif regularization == 1:
                        delta_w = np.sum(error * x[start : end, :], axis=0) + lam * np.sign(self.w)
                        delta_b = np.sum(error) + lam * np.sign(self.b)
                    elif regularization == 2:
                        delta_w = np.sum(error * x[start : end, :], axis=0) + 2 * lam * self.w
                        delta_b = np.sum(error) + 2 * lam * self.b
                    else:
                        raise Error("invalid regularization!")
                    self.w -= self.eta * delta_w
                    self.b -= self.eta * delta_b
                if detailed:
                    print("epoch:", i, " w:", self.w, " b:", self.b)

    def predict(self, x):
        return np.sum(self.w * x, axis=-1) + self.b


class LW_LinearRegression:
    """加权线性回归"""
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x, y, use_normal_equation=True, eta=0.1, detailed=False):
        self.x = x.copy()
        self.y = np.reshape(y.copy(), [-1, 1])
        self.use_normal_equation = use_normal_equation

    def predict(self, x, k=1.0):
        N = (self.x).shape[0]
        X = np.hstack((np.ones([N, 1]), self.x))
        weights = np.eye(N)
        if self.use_normal_equation:
            if x.ndim == 1:
                x = np.insert(x, obj=0, values=1)
                for i in range(N):
                    tmp = np.sum((x - X[i, :]) ** 2)
                    weights[i, i] = np.exp(tmp / (-2.0 * k ** 2))

                inv_matrix = np.linalg.inv(np.dot(np.dot(X.T, weights), X))
                w = np.dot(np.dot(inv_matrix, X.T), np.dot(weights, self.y))
                w = w.flatten()
                return np.sum(x * w)
            else:
                return np.array([self.predict(i, k=k) for i in x])