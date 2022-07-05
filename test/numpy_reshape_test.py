import numpy as np
X = np.arange(576)
print("X.shape: ", X.shape)

SHAPE =  int(np.sqrt(X.shape[0]))
print(f"SHAPE: {SHAPE}")

X_1 = X.reshape(1,-1)
print("X_1.shape: ", X_1.shape)

X_1_1 = X.reshape(-1,1)
print("X_1_1.shape: ", X_1_1.shape)

X_1_2 = X.reshape(-1,1,1)
print("X_1_2.shape: ", X_1_2.shape)

X_2 = X.reshape(SHAPE,SHAPE)
print("X_2.shape: ", X_2.shape)

X_3 = X.reshape(SHAPE,SHAPE, 1)
print("X_3.shape: ", X_3.shape)
print()
