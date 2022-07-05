import numpy as np
from sklearn import datasets, cluster


digits = datasets.load_digits()
images = digits.images
print(f"images.shape: {images.shape}")

X = np.reshape(images, (len(images), -1))
print(f"X.shape: {X.shape}")

agglo = cluster.FeatureAgglomeration(n_clusters=32)
agglo.fit(X) 
# FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto',
#            connectivity=None, linkage='ward', memory=None, n_clusters=32,
#            pooling_func=...)

X_reduced = agglo.transform(X)
print(f"X_reduced.shape: {X_reduced.shape}")