import numpy as np
import model.perceptron.perceptron_dual as pd

p = pd.Perceptron(1)

X = np.array([
    [3,3],
    [4,3],
    [1,1]
])

print(pd.gram(X))

y = np.array([1,1,-1])

p.fit(X, y, True)

print(p.w, p.b)

