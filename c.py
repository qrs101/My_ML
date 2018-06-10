import numpy as np

a = np.array([1,2,3])
b = np.array([2,3,4])

X = np.array([
    [3,3],
    [4,3],
    [1,1]
])

#print(np.dot(a, b))
print(a[..., None] * X)