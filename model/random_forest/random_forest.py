import numpy as np
from model.error import Error
from model.decision_tree.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, input_attr=0, output_attr=0, num=20, feature_sampling=np.log2):
        if input_attr != 0 and input_attr != 1:
            raise Error("Invalid input_attr!")
        if output_attr != 0 and output_attr != 1:
            raise Error("Invalid output_attr!")

        self.num = num
        self.input_attr = input_attr
        self.output_attr = output_attr
        self.feature_sampling = feature_sampling
        self.trees = list()
        for i in range(num):
            tree = DecisionTree(input_attr, output_attr, type="CART")
            self.trees.append(tree)

    def bootstrap(self, x, y):
        sampling_x, sampling_y = list(), list()
        n = x.shape[0]
        for i in range(n):
            index = np.random.randint(n)
            sampling_x.append(x[index])
            sampling_y.append(y[index])
        return np.array(sampling_x), np.array(sampling_y)

    def fit(self, x, y):
        for i in range(self.num):
            sampling_x, sampling_y = self.bootstrap(x, y)
            self.trees[i].fit(sampling_x, sampling_y, self.feature_sampling)

    def predict(self, x):
        res = list()
        for i in range(self.num):
            res.append(self.trees[i].predict(x))
        res = np.array(res)
        if self.output_attr == 1:
            return np.sum(res) / len(res)
        else:
            labels, cnts = np.unique(res, return_counts=True)
            label, max_cnt = None, 0
            for i in range(len(labels)):
                if max_cnt < cnts[i]:
                    max_cnt = cnts[i]
                    label = labels[i]
            return label