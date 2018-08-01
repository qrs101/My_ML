import numpy as np
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

if __name__ == "__main__":
    x, y = load_data()
    c = dt.DecisionTree()
    c.fit(x, y)

