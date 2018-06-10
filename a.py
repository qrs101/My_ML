import numpy as np
import model.perceptron.perceptron_primary as mp
import model.perceptron.perceptron_dual as pd

p = mp.Perceptron(1)

X = np.array([
    [3,3],
    [4,3],
    [1,1]
])

y = np.array([1,1,-1])

p.fit(X, y, True)

print(p.w, p.b)

