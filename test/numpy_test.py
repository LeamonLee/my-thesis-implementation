import numpy as np
import pandas as pd

a = [[1.1, 2.22],
        [3.334, 4.444]]

a = np.array(a)
print("a.shape: ", a.shape)
a = a.reshape(a.shape[0], a.shape[1], 1)
print("a.shape: ", a.shape)

# ======================================

b = [[5.5, 6.66],
        [7.777, 8.888]]
b = np.array(b)
b = b.reshape(b.shape[0], b.shape[1], 1)

c = np.concatenate((a,b), axis=2)
print("c: ", c)
print("c.shape: ", c.shape)
# ======================================

fileName = "numpy_test.npy"
np.save(fileName, a)
b = np.load(fileName)

print(b)
print(b.shape)
print(b.dtype)

if (a == b).all():
    print("True")
else:
    print("False")

# ======================================

mat1 = pd.DataFrame({"col1": [1,2,3,4,5], "col2": [None, 20, np.inf, 40, None], "col3": [100, None, 300, 400, 500]})
print(f"mat1: {mat1}")
print(f"np.any(np.isnan(mat1)): {np.any(np.isnan(mat1))}")
print(f"np.all(np.isfinite(mat1)): {np.all(np.isfinite(mat1))}")

mat2 = pd.DataFrame({"col1": [1,2,3,4,5], "col2": [10, 20, 30, 40, 50], "col3": [100, 200, 300, 400, 500]})
print(f"mat2: {mat2}")
print(f"np.any(np.isnan(mat2)): {np.any(np.isnan(mat2))}")
print(f"np.all(np.isfinite(mat2)): {np.all(np.isfinite(mat2))}")