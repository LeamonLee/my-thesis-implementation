from sklearn.cluster import FeatureAgglomeration
import pandas as pd
import matplotlib.pyplot as plt

# 使用uci的dataset
#iris.data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
iris=pd.read_csv('iris.data',sep=',',header=None)
print(f"iris.head(): {iris.head()}")

# 使用sklearn內建的dataset
# from sklearn.datasets import load_iris
# iris = load_iris()

# x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
# y = pd.DataFrame(iris["target"], columns=["target_name"])
# print(f"x.head(): {x.head()}")
# print(f"y.head(): {y.head()}")


y_label = iris[4]
x_iris = iris.drop([4],axis=1)
print(f"type(x_iris): {type(x_iris)}")
print(f"x_iris.head(): {x_iris.head()}")
print(f"y_label.head(): {y_label.head()}")

# set n_clusters to 2, the output will be two columns of agglomerated features ( iris has 4 features)
# agglo=FeatureAgglomeration(n_clusters=2).fit_transform(x_iris)    # 如果寫這行，agglo.labels_就印不出東西

agglo = FeatureAgglomeration(n_clusters=2, compute_distances=True)
# agglo = FeatureAgglomeration(n_clusters=2, affinity='precomputed', linkage="average", compute_distances=True)
x_iris_tranformed = agglo.fit_transform(x_iris)

print(f"agglo.distances_: {agglo.distances_}")

# After fitting the clusterer, 
# agglo.labels_ contains a list that tells
# in which cluster in the reduced dataset each feature in the original dataset belongs.
print(f"agglo.labels_: {agglo.labels_}")
for i, label in enumerate(set(agglo.labels_)):
    features_with_label = [j for j, lab in enumerate(agglo.labels_) if lab == label]
    print('Features in agglomeration {}: {}'.format(i, features_with_label))

print(f"x_iris_tranformed: {x_iris_tranformed}")
# print(f"x_iris_tranformed[:,0]: {x_iris_tranformed[:,0]}")
# print(f"x_iris_tranformed[:,1]: {x_iris_tranformed[:,1]}")

# scatter plotting
color=[]
for i in y_label:
    if i=='Iris-setosa':
        color.append('g')
    if  i=='Iris-versicolor':
        color.append('b')
    if i=='Iris-virginica':
        color.append('r')
plt.scatter(x_iris_tranformed[:,0],x_iris_tranformed[:,1],c=color)
plt.show()