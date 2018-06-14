import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, eta, batch_size = 1, epoch = 1000):
        """
        :param eta:         学习率
        :param batch_size:  一个batch中样本数量
        :param epoch:       迭代次数
        """
        self.w = None
        self.b = None
        self.eta = eta
        self.batch_size = batch_size
        self.epoch = epoch

    def fit(self, X, y, detailed = False):
        """
        :param X:
        :param y:
        :param detailed:
        :return:
        """
        if X.shape[0] != y.shape[0]:
            print("输入格式错误")
            return

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        self.w = np.zeros(X.shape[1])
        self.b = np.zeros(1)
        num_of_batch = X.shape[0] // self.batch_size
        if X.shape[0] % self.batch_size != 0:
            num_of_batch += 1

        for i in range(self.epoch):
            for j in range(num_of_batch):
                start = j * self.batch_size
                end = min((j + 1) * self.batch_size, X.shape[0])
                error = self.predict(X[start : end, :]) - y[start : end]
                error = error.reshape([-1, 1])
                delta_w = np.sum(error * X[start : end, :], axis = 0)
                delta_b = np.sum(error)
                self.w -= self.eta * delta_w
                self.b -= self.eta * delta_b
            if detailed:
                print("epoch:", i, " w:", self.w, " b:", self.b)

    def predict(self, x):
        return sigmoid(np.sum(self.w * x, axis = -1) + self.b)

