import pandas as pd
from varclushi import VarClusHi

iris=pd.read_csv('iris.data',sep=',',header=None)
print(f"iris.head(): {iris.head()}")
y_label = iris[4]
x_iris = iris.drop([4],axis=1)

demo1_vc = VarClusHi(x_iris, maxeigval2=1, maxclus=None)
demo1_vc.varclus()

print(f"demo1_vc.info: {demo1_vc.info}")
print()
print(f"demo1_vc.rsquare: {demo1_vc.rsquare}")