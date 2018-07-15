import os
import numpy as np
import model.knn.knn as knn

def load_data_1():
    filename = "../dataset/datingTestSet2.txt"
    file = open(filename)
    x, y = list(), list()
    for line in file.readlines():
        curLine = line.strip().split('\t')
        tmp = list()
        for i in range(3):
            print(curLine[i])
            print(type(curLine[i]))
            tmp.append(float(curLine[i]))
        x.append(tmp)
        y.append(int(curLine[3]))

    return np.array(x), np.array(y)

def load_data_2():
    train_path = "../dataset/digits/trainingDigits/"
    test_path = "../dataset/digits/testDigits/"
    train_x, train_y, test_x, test_y = list(), list(), list(), list()
    for filename in os.listdir(train_path):
        file = open(train_path + filename)
        tmp = list()
        for number in file.read():
            if number == '\n':
                continue
            #print(number)
            #print(type(number))
            tmp.append(int(number))
        train_x.append(tmp)
        train_y.append(int(filename[0]))

    for filename in os.listdir(test_path):
        file = open(test_path + filename)
        tmp = list()
        for number in file.read():
            if number == '\n':
                continue
            tmp.append(int(number))
        test_x.append(tmp)
        test_y.append(int(filename[0]))

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def norm(x):
    min = x.min(axis=0)
    max = x.max(axis=0)
    norm_x = (x - min) / (max - min)
    return norm_x

def example_1():
    x, y = load_data_1()
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

def example_2():
    train_x, train_y, test_x, test_y = load_data_2()

    c = knn.knn()
    c.fit(train_x, train_y, use_kd_tree=False)

    err = 0
    for i in range(len(test_x)):
        predict_y = c.predict(test_x[i], k=5, detailed=False)
        if predict_y != test_y[i]:
            err += 1

    r = err / len(test_y)
    print(r)

if __name__ == "__main__":
    example_1()
    #example_2()
