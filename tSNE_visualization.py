from pymongo import MongoClient
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import util

NUM_PCA_COMP = 50
client = MongoClient()
db = client['image_features']
coll = db['features']

cursor = coll.find({})
clusters_df = pd.DataFrame(list(cursor))

centroids = clusters_df['sparse_features'].apply(lambda x : util.sparse_to_numpy(x))

build_mat = []
for row in centroids:
    build_mat.append(list(row))
new_centroids = np.array(build_mat)

pca = PCA(n_components=NUM_PCA_COMP)
X = pca.fit_transform(new_centroids)
print "Shape of the centroids after dimensionality reduction: ", X.shape

stochastic_neighbors = TSNE(n_components=2, random_state=0)
image_centers = stochastic_neighbors.fit_transform(X)
np.save('tSNE_centers', image_centers)
