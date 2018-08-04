import numpy as np
import random
import model.decision_tree.decision_tree as dt

def load_data():
    filename = "../dataset/lenses.txt"
    file = open(filename)
    x, y = list(), list()
    for line in file.readlines():
        curLine = line.strip().split('\t')
        tmp = list()
        for i in range(4):
            tmp.append(curLine[i])
        x.append(tmp)
        y.append(curLine[-1])

    return np.array(x), np.array(y)

def example_1():
    x, y = load_data()
    c = dt.DecisionTree()
    c.fit(x, y)

def example_2():
    from sklearn import datasets

    iris = datasets.load_iris()  # 加载 iris 数据集
    x = iris.data
    y = iris.target
    train = list(range(150))
    test, all = [], 30
    for i in range(all):
        r = random.randint(0, len(train) - 1)
        test.append(train[r])
        train.remove(train[r])

    c = dt.DecisionTree(type="CART", input_attr=1, output_attr=0)
    c.fit(x[train], y[train])
    err = 0
    for i in test:
        res = c.predict(x[i])
        # print(res, y[i])
        if res != y[i]:
            err += 1

    print(err / all)

if __name__ == "__main__":
    example_2()