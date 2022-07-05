import pandas as pd
from sklearn.utils import shuffle

f = pd.read_csv('Students2.csv')
X_df = f.iloc[:, :-1]

print(f"X_df: {X_df.head()}")
print(f"X_df.index: {X_df.index}")
X_shuffled_df = shuffle(X_df)
print(f"X_shuffled_df: {X_shuffled_df.head()}")
print(f"X_shuffled_df.index: {X_shuffled_df.index}")