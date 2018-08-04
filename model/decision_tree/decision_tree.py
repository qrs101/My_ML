import numpy as np
from model.error import Error

def entropy(data):
    _, cnt = np.unique(data, return_counts=True)
    prob = cnt / len(data)
    return -np.sum(prob * np.log2(prob))

def calculate_info_gain(col_x, y):
    # for ID3
    H_y = entropy(y)
    val_x, cnt_x = np.unique(col_x, return_counts=True)
    prob_x = cnt_x / len(col_x)
    H_y_given_x = 0
    for i in range(len(val_x)):
        H_y_given_x += prob_x[i] * entropy(y[col_x == val_x[i]])
    return H_y - H_y_given_x, val_x

def calculate_info_gain_ratio(col_x, y):
    # for C4.5
    H_y = entropy(y)
    val_x, cnt_x = np.unique(col_x, return_counts=True)
    prob_x = cnt_x / len(col_x)
    H_y_given_x = 0
    for i in range(len(val_x)):
        H_y_given_x += prob_x[i] * entropy(y[col_x == val_x[i]])
    return (H_y - H_y_given_x) / entropy(col_x), val_x

def Gini(data):
    # for Classification Tree
    _, cnt = np.unique(data, return_counts=True)
    prob = cnt / len(data)
    return 1 - np.sum(prob ** 2)

def MSE(data):
    # for Regression Tree
    return np.var(data)

def choose_best_value(col_x, y, input_attr=0, err_func=None):
    split_value = None
    min_error = np.inf
    if input_attr == 1:
        sorted_index = np.argsort(col_x)
        for i in range(1, len(sorted_index)):
            index1, index2 = sorted_index[i - 1], sorted_index[i]
            value = (col_x[index1] + col_x[index2]) / 2
            post_error = err_func(y[col_x <= value]) + err_func(y[col_x > value])
            if min_error > post_error:
                min_error = post_error
                split_value = value
    else:
        unique_x = np.unique(col_x)
        for value in unique_x:
            post_error = err_func(y[col_x == value]) + err_func(y[col_x != value])
            if min_error > post_error:
                min_error = post_error
                split_value = value

    return split_value, min_error


def choose_best_feature(x, y, input_attr=0, output_attr=0):
    """for CART
    :param x:              输入
    :param y:              输出
    :param input_attr:     输入属性，1：连续，0：离散
    :param output_attr:    输出属性，1：连续，0：离散
    :return:               最佳切分特征和最佳切分点

    ## 输出属性连续，表示回归问题，采用均方差度量损失
    ## 输出属性离散，表示分类问题，采用基尼指数度量损失

    ## 输入属性连续，则对输入排序后，以任意两个相邻点中位数作为切分点，选取最优切分点，将数据划分为两部分
    ## 输入属性离散，则依次遍历每个可能取值，以是否等于改值为标准，将数据集划分为两部分，选取最优特征取值
    """
    if input_attr != 0 and input_attr != 1:
        raise Error("Invalid input_attr!")
    if output_attr != 0 and output_attr != 1:
        raise Error("Invalid output_attr!")

    if output_attr == 1:
        err_func = MSE
    else:
        err_func = Gini

    min_error = np.inf
    split_feature, split_value = None, None
    for i in range(x.shape[1]):
        value, error = choose_best_value(x[:, i], y, input_attr=input_attr, err_func=err_func)
        if min_error > error:
            min_error = error
            split_feature = i
            split_value = value

    return split_feature, split_value, err_func(y) - min_error

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
    def __init__(self, type="ID3", input_attr=0, output_attr=0, pre_pruning=(0, 0)):
        self.root = None
        self.type = type
        self.input_attr = input_attr
        self.output_attr = output_attr
        self.pre_pruning = pre_pruning

    def is_same_kind(self, y):
        val, cnt = np.unique(y, return_counts=True)
        if len(val) == 1:
            return True, val[0]
        max_cnt, index = 0, 0
        for i in range(len(cnt)):
            if max_cnt < cnt[i]:
                max_cnt = cnt[i]
                index = i
        return False, val[index]

    def build_tree(self, x, y, features):
        if len(y) < self.pre_pruning[1]:
            return None

        is_same_kind, label = self.is_same_kind(y)
        if is_same_kind or len(features) == 0:
            return Node(label, is_leaf=True)

        split_axis, split_values, max_info = None, None, 0
        for feature in features:
            if self.type == "ID3":
                info, values = calculate_info_gain(x[:, feature], y)
            elif self.type == "C4.5":
                info, values = calculate_info_gain_ratio(x[:, feature], y)
            else:
                raise Error("Invalid type!")
            if max_info < info:
                split_axis = feature
                split_values = values
                max_info = info

        if max_info < self.pre_pruning[0]:
            return Node(label, is_leaf=True)

        node = Node(label, is_leaf=False)
        for value in split_values:
            new_features = features.copy()
            new_features.remove(split_axis)
            child = self.build_tree(x[x[:, split_axis] == value], y[x[:, split_axis] == value], new_features)
            if child is None:
                return Node(label, is_leaf=True)
            node.add_child(value, child)

        return node

    def build_CART_tree(self, x, y):
        if len(y) < self.pre_pruning[1]:
            return None

        is_same_kind, label = self.is_same_kind(y)
        if is_same_kind:
            return Node(label, is_leaf=True)

        axis, value, err_var = choose_best_feature(x, y, self.input_attr, self.output_attr)
        if err_var < self.pre_pruning[0]:
            return Node(label, is_leaf=True)

        node = Node(label, split_axis=axis, split_value=value, is_leaf=False)
        if self.input_attr == 1:
            le_child = self.build_CART_tree(x[x[:, axis] <= value], y[x[:, axis] <= value])
            ri_child = self.build_CART_tree(x[x[:, axis] > value], y[x[:, axis] > value])
            if le_child is None or ri_child is None:
                return Node(label, is_leaf=True)
        else:
            le_child = self.build_CART_tree(x[x[:, axis] == value], y[x[:, axis] == value])
            ri_child = self.build_CART_tree(x[x[:, axis] != value], y[x[:, axis] != value])
            if le_child is None or ri_child is None:
                return Node(label, is_leaf=True)
        node.add_child(True, le_child)
        node.add_child(False, ri_child)

        return node

    def fit(self, x, y):
        if x.shape[0] != y.shape[0] or len(y) == 0:
            raise Error("Invalid x, y!")

        if self.type == "CART":
            self.root = self.build_CART_tree(x, y)
        else:
            features = list(range(x.shape[1]))
            self.root = self.build_tree(x, y, features)

    #To_Do
    def pruning(self):
        pass

    def predict(self, x):
        node = self.root
        if self.type == "CART":
            if self.input_attr == 1:
                while not node.is_leaf:
                    node = node.child[x[node.split_axis] <= node.split_value]
            else:
                while not node.is_leaf:
                    node = node.child[x[node.split_axis] == node.split_value]
        else:
            while not node.is_leaf:
                node = node.child[x[node.split_axis]]

        return node.label