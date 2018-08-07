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

def choose_best_value(col_x, y, err_func, input_attr=0):
    # for CART
    split_value = None
    min_error = np.inf
    if input_attr == 1:
        sorted_index = np.argsort(col_x)
        for i in range(1, len(sorted_index)):
            index1, index2 = sorted_index[i - 1], sorted_index[i]
            value = (col_x[index1] + col_x[index2]) / 2
            error = err_func(y[col_x <= value]) + err_func(y[col_x > value])
            if min_error > error:
                min_error = error
                split_value = value
    else:
        unique_x = np.unique(col_x)
        for value in unique_x:
            error = err_func(y[col_x == value]) + err_func(y[col_x != value])
            if min_error > error:
                min_error = error
                split_value = value

    return split_value, min_error

def choose_best_feature(x, y, input_attr=0, output_attr=0, features=None):
    # for CART
    if output_attr == 1:
        err_func = MSE
    else:
        err_func = Gini

    if features is None:
        features = np.arange(x.shape[1])

    min_error = np.inf
    split_feature, split_value = None, None
    for feature in features:
        value, error = choose_best_value(x[:, feature], y, err_func=err_func, input_attr=input_attr)
        if min_error > error:
            min_error = error
            split_feature = feature
            split_value = value

    return split_feature, split_value, err_func(y) - min_error

class Node:
    def __init__(self, label, split_feature=None, split_value=None, is_leaf=True):
        self.label = label
        self.split_feature = split_feature
        self.split_value = split_value        #for CART
        self.is_leaf = is_leaf
        self.child = dict()

    def __str__(self):
        res = str(self.split_feature)
        return res

    def add_child(self, value, node):
        self.child[value] = node

class DecisionTree:
    def __init__(self, input_attr=0, output_attr=0, type="CART", pre_pruning=(0, 0)):
        """
        :param input_attr:      输入属性，1：连续，0：离散   for CART
        :param output_attr:     输出属性，1：连续，0：离散   for CART
        :param type:            ID3, C4.5, CART
        :param pre_pruning:     2元组，第一个元素表示损失变化的最小值，第二个元素表示节点上最小样本数量

        ## 输出属性连续，表示回归问题，采用均方差度量损失
        ## 输出属性离散，表示分类问题，采用基尼指数度量损失

        ## 输入属性连续，则对输入排序后，以任意两个相邻点中位数作为切分点，选取最优切分点，将数据划分为两部分
        ## 输入属性离散，则依次遍历每个可能取值，以是否等于改值为标准，将数据集划分为两部分，选取最优特征取值
        """
        self.root = None
        self.type = type
        self.input_attr = input_attr
        self.output_attr = output_attr
        self.pre_pruning = pre_pruning
        if input_attr != 0 and input_attr != 1:
            raise Error("Invalid input_attr!")
        if output_attr != 0 and output_attr != 1:
            raise Error("Invalid output_attr!")

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

        split_feature, split_values, max_info = None, None, 0
        for feature in features:
            if self.type == "ID3":
                info, values = calculate_info_gain(x[:, feature], y)
            elif self.type == "C4.5":
                info, values = calculate_info_gain_ratio(x[:, feature], y)
            else:
                raise Error("Invalid type!")
            if max_info < info:
                split_feature = feature
                split_values = values
                max_info = info

        if max_info < self.pre_pruning[0]:
            return Node(label, is_leaf=True)

        node = Node(label, is_leaf=False)
        for value in split_values:
            new_features = np.delete(features, split_feature)
            value_x = x[x[:, split_feature] == value]
            value_y = y[x[:, split_feature] == value]
            child = self.build_tree(value_x, value_y, new_features)
            if child is None:
                return Node(label, is_leaf=True)
            node.add_child(value, child)

        return node

    def build_CART_tree(self, x, y, feature_sampling=None):
        if len(y) < self.pre_pruning[1]:
            return None

        if self.output_attr == 1:
            label = np.sum(y) / len(y)
        else:
            is_same_kind, label = self.is_same_kind(y)
            if is_same_kind:
                return Node(label, is_leaf=True)

        features = np.arange(x.shape[1])
        if feature_sampling is not None:
            if isinstance(feature_sampling, int):
                features = np.random.choice(features, feature_sampling)
            else:
                features = np.random.choice(features, feature_sampling(len(features)))

        parameter = (x, y, self.input_attr, self.output_attr, features)
        split_feature, split_value, err_var = choose_best_feature(*parameter)

        if err_var < self.pre_pruning[0]:
            return Node(label, is_leaf=True)

        node = Node(label, split_feature=split_feature, split_value=split_value, is_leaf=False)
        if self.input_attr == 1:
            le_x = x[x[:, split_feature] <= split_value]
            le_y = y[x[:, split_feature] <= split_value]
            ri_x = x[x[:, split_feature] > split_value]
            ri_y = y[x[:, split_feature] > split_value]
            le_child = self.build_CART_tree(le_x, le_y, feature_sampling)
            ri_child = self.build_CART_tree(ri_x, ri_y, feature_sampling)
            if le_child is None or ri_child is None:
                return Node(label, is_leaf=True)
        else:
            le_x = x[x[:, split_feature] == split_value]
            le_y = y[x[:, split_feature] == split_value]
            ri_x = x[x[:, split_feature] != split_value]
            ri_y = y[x[:, split_feature] != split_value]
            le_child = self.build_CART_tree(le_x, le_y, feature_sampling)
            ri_child = self.build_CART_tree(ri_x, ri_y, feature_sampling)
            if le_child is None or ri_child is None:
                return Node(label, is_leaf=True)

        node.add_child(True, le_child)
        node.add_child(False, ri_child)

        return node

    def fit(self, x, y, feature_sampling=None):
        if x.shape[0] != y.shape[0] or len(y) == 0:
            raise Error("Invalid x, y!")

        if self.type == "CART":
            self.root = self.build_CART_tree(x, y, feature_sampling)
        elif self.type == "ID3" or self.type == "C4.5":
            self.root = self.build_tree(x, y, np.arange(x.shape[1]))
        else:
            raise Error("Invalid type!")

    def pruning(self):
        # To_Do
        pass

    def predict(self, x):
        node = self.root
        if self.type == "CART":
            if self.input_attr == 1:
                while not node.is_leaf:
                    node = node.child[x[node.split_feature] <= node.split_value]
            else:
                while not node.is_leaf:
                    node = node.child[x[node.split_feature] == node.split_value]
        else:
            while not node.is_leaf:
                node = node.child[x[node.split_feature]]

        return node.label