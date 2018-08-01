import numpy as np
import model.logistic_regression.logistic_regression as lr

def loadData():
    train_file = open("../dataset/horseColicTraining.txt")
    test_file = open("../dataset/horseColicTest.txt")
    train_X, train_y = [], []
    for line in train_file.readlines():
        curLine = line.strip().split('\t')
        l = []
        for i in range(21):
            l.append(float(curLine[i]))
        train_X.append(l)
        train_y.append(float(curLine[21]))

    test_X, test_y = [], []
    for line in test_file.readlines():
        curLine = line.strip().split('\t')
        l = []
        for i in range(21):
            l.append(float(curLine[i]))
        test_X.append(l)
        test_y.append(float(curLine[21]))

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

def run():
    train_X, train_y, test_X, test_y = loadData()
    classifier = lr.LogisticRegression(0.01, 400, None)
    classifier.fit(train_X, train_y)

    error, total = 0, 0
    for i in range(test_X.shape[0]):
        if int(classifier.predict(test_X[i])) != int(test_y[i]):
            error += 1
        total += 1

    print(error / total)

if __name__ == "__main__":
    run()

