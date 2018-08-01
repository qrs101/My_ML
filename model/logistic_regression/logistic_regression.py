import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, eta=0.1, epoch=1000, batch_size=1):
        """
        :param eta:         学习率
        :param epoch:       迭代次数
        :param batch_size:  batch size, 默认随机梯度下降, 如果为None则采用全部训练集
        """
        self.w = None
        self.b = None
        self.eta = eta
        self.epoch = epoch
        self.batch_size = batch_size

    def fit(self, X, y, detailed=False):
        """
        :param X:         输入矩阵, 2维numpy数组
        :param y:         输入标签, 1维numpy数组
        :param detailed:  输出迭代详细信息
        :return:
        """
        if X.shape[0] != y.shape[0]:
            print("输入格式错误")
            return

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        self.w = np.ones(X.shape[1])
        self.b = np.ones(1)
        num_of_batch = X.shape[0] // self.batch_size
        if X.shape[0] % self.batch_size != 0:
            num_of_batch += 1

        for i in range(self.epoch):
            for j in range(num_of_batch):
                start = j * self.batch_size
                end = min((j + 1) * self.batch_size, X.shape[0])
                error = self.probability(X[start : end, :]) - y[start : end]
                error = error.reshape([-1, 1])
                delta_w = np.sum(error * X[start : end, :], axis = 0)
                delta_b = np.sum(error)
                self.w -= self.eta * delta_w
                self.b -= self.eta * delta_b
            if detailed:
                print("epoch:", i, " w:", self.w, " b:", self.b)

    def probability(self, x):
        return sigmoid(np.sum(self.w * x, axis=-1) + self.b)

    def predict(self, x):
        if self.probability(x) > 0.5:
            return 1
        else:
            return 0

