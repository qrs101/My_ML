import numpy as np
import model.knn.knn as knn

def load_data():
    filename = "../dataset/datingTestSet2.txt"
    file = open(filename)
    x, y = list(), list()
    for line in file.readlines():
        curLine = line.strip().split('\t')
        l = list()
        for i in range(3):
            l.append(float(curLine[i]))
        x.append(l)
        y.append(int(curLine[3]))

    return np.array(x), np.array(y)

def norm(x):
    min = x.min(axis=0)
    max = x.max(axis=0)
    norm_x = (x - min) / (max - min)
    return norm_x

if __name__ == "__main__":
    x, y = load_data()
    x = norm(x)
    m = int(0.1 * x.shape[0])

    c1 = knn.knn()
    c1.fit(x[m:], y[m:], use_kd_tree=False)
    c2 = knn.knn()
    c2.fit(x[m:], y[m:], use_kd_tree=True)

    err1, err2 = 0, 0
    for i in range(m):
        y1 = c1.predict(x[i], k=3, detailed=True)
        y2 = c2.predict(x[i], k=3, detailed=True)
        if y1 != y[i]:
            err1 += 1
        if y2 != y[i]:
            err2 += 1

    r1, r2 = err1 / m, err2 / m
    print(r1, r2)
