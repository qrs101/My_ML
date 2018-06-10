import numpy as np

def gram(X):
    #return np.matmul(X, X.T)
    n = X.shape[0]
    G = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.dot(X[i], X[j])
            G[j, i] = G[i, j]
    return G

class Perceptron:
    def __init__(self, eta, max_epoch = 1000):
        """
        :param eta: 学习率
        :param max_epoch: 最大迭代次数
        """
        self.eta = eta
        self.max_epoch = max_epoch

    def fit(self, X, y, detailed = False):
        """
        :param X: 输入矩阵
        :param y: 输入标签
        :param detailed: 输出每一步详细信息
        :return:
        """
        if X.shape[0] != y.shape[0]:
            print("输入格式错误")
            return

        self.alpha = np.zeros(X.shape[0])
        self.b = np.zeros(1)
        G = gram(X)

        i = 0
        cnt = 0
        epoch = 0

        while cnt != X.shape[0] and (self.max_epoch is None or epoch <= self.max_epoch):
            cnt += 1

            if y[i] * (np.sum(self.alpha * y * G[i]) + self.b) <= 0:
                cnt = 0
                epoch += 1
                self.alpha[i] += self.eta
                self.b += self.eta * y[i]
                if detailed:
                    print("epoch:", epoch, " alpha:", self.alpha, " b:", self.b)

            i = (i + 1) % X.shape[0]

        tmp = y * self.alpha
        tmp = tmp.reshape([-1, 1])

        self.w = np.sum(tmp * X, axis = 0)

    def preditc(self, x):
        return np.sign(np.dot(self.w, x) + self.b)