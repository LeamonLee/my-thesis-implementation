import pandas as pd

a = pd.DataFrame({"col1": [1,2,3], "col2": [10,20,30], "col3": [100,200,300]})

print(a.values)
print()
print(a.to_numpy())