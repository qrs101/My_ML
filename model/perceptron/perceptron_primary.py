import numpy as np

class Perceptron:
    def __init__(self, eta, max_epoch = 1000):
        """
        :param eta:        学习率
        :param max_epoch:  最大迭代次数
        """
        self.w = None
        self.b = None
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

        self.w = np.zeros(X.shape[1])
        self.b = np.zeros(1)
        i = 0
        cnt = 0
        epoch = 0

        while cnt != X.shape[0] and (self.max_epoch is None or epoch <= self.max_epoch):
            cnt += 1

            if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                cnt = 0
                epoch += 1
                self.w += self.eta * y[i] * X[i]
                self.b += self.eta * y[i]
                if detailed:
                    print("epoch:", epoch, " w:", self.w, " b:", self.b)

            i = (i + 1) % X.shape[0]

    def predict(self, x):
        return np.sign(np.dot(self.w, x) + self.b)