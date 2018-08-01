import numpy as np
from model.error import Error

def entropy(data):
    _, cnt = np.unique(data, return_counts=True)
    prob = cnt / len(data)
    return -np.sum(prob * np.log2(prob))

def calculate_info_gain(col_x, y):
    H_y = entropy(y)
    val_x, cnt_x = np.unique(col_x, return_counts=True)
    prob_x = cnt_x / len(col_x)
    H_y_given_x = 0
    for i in range(len(val_x)):
        H_y_given_x += prob_x[i] * entropy(y[col_x == val_x[i]])
    return H_y - H_y_given_x, val_x

def calculate_info_gain_ratio(col_x, y):
    H_y = entropy(y)
    val_x, cnt_x = np.unique(col_x, return_counts=True)
    prob_x = cnt_x / len(col_x)
    H_y_given_x = 0
    for i in range(len(val_x)):
        H_y_given_x += prob_x[i] * entropy(y[col_x == val_x[i]])
    return (H_y - H_y_given_x) / entropy(col_x), val_x

def Gini(data):
    _, cnt = np.unique(data, return_counts=True)
    prob = cnt / len(data)
    return 1 - np.sum(prob ** 2)

class Node:
    def __init__(self, label, split_axis=None, split_value=None, is_leaf=True):
        self.label = label
        self.split_axis = split_axis
        self.split_value = split_value
        self.is_leaf = is_leaf
        self.child = dict()

    def __str__(self):
        res = str(self.split_axis)
        #for k, v in self.child.items():
        #    res += " " + str(k) + ":" + str(v) + " "
        #res = res + "{" +
        return res

    def add_child(self, value, node):
        self.child[value] = node

class DecisionTree:
    def __init__(self, epsilon=0, type="ID3"):
        self.root = None
        self.type = type
        self.epsilon = epsilon

    def is_same_class(self, labels):
        val, cnt = np.unique(labels, return_counts=True)
        if len(val) == 1:
            return True, val[0]
        max_cnt, index = 0, 0
        for i in range(len(cnt)):
            if max_cnt < cnt[i]:
                max_cnt = cnt[i]
                index = i
        return False, val[index]

    def build_tree(self, x, y, features):
        is_same_class, label = self.is_same_class(y)
        if is_same_class or len(features) == 0:
            return Node(label, is_leaf=True)

        split_axis, val_of_x, max_info = None, None, 0
        for feature in features:
            if self.type == "ID3":
                info, val = calculate_info_gain(x[:, feature], y)
            elif self.type == "C4.5":
                info, val = calculate_info_gain_ratio(x[:, feature], y)
            else:
                raise Error("Invalid type!")
            if max_info <= info:
                split_axis = feature
                val_of_x = val
                max_info = info

        if max_info < self.epsilon:
            return Node(label, is_leaf=True)

        node = Node(label, split_axis=split_axis, is_leaf=True)
        for val in val_of_x:
            new_features = features.copy()
            new_features.remove(split_axis)
            child = self.build_tree(x[x[:, split_axis] == val], y[x[:, split_axis] == val], new_features)
            if child is not None:
                node.is_leaf = False
                node.add_child(val, child)

        return node

    def fit(self, x, y):
        if x.shape[0] != y.shape[0] or len(y) == 0:
            raise Error("Invalid inputs!")
        features = list(range(x.shape[1]))
        if self.type == "CART":
            pass
        else:
            self.root = self.build_tree(x, y, features)

    #To_Do
    def pruning(self):
        pass

    def predict(self, x):
        node = self.root
        while not node.is_leaf:
            node = node.child[x[node.split_axis]]

        return node.label



