import numpy as np
import matplotlib.pyplot as plt
import model.linear_regression.linear_regression as lr

def load_data():
    file = open("../dataset/abalone.txt")
    x, y = [], []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        tmp = list()
        for i in range(len(curLine) - 1):
            tmp.append(float(curLine[i]))
        x.append(tmp)
        y.append(float(curLine[-1]))

    return np.array(x), np.array(y)

if __name__ == "__main__":
    x, y = load_data()
    r = lr.LW_LinearRegression()
    r.fit(x[0:99], y[0:99])
    preditct_y = r.predict(x[0:99], k=0.1)
    print(y)
    print(preditct_y)