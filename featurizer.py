import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from building_dataset import *
from util import create_doc, inner_prod, sparcify
from ConvNeuralNet import ConvNeuralNet
from pymongo import MongoClient
from theano import function
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from collections import defaultdict
import sys
import os
import cPickle

# Featurizes the imput images and writes the results into mongoDB collections
PICKLED_MODEL_FOLDER = 'vgg_cnn_s.pkl'
NUM_SAMPLES = 25000
NUM_CLUSTERS = 100

# Load the convolutional neural net's weights
cnn = ConvNeuralNet()
cnn.load()
client = MongoClient()

# Create a database
db = client['image_features']
result_1 = db.features.delete_many({})
result_2 = db.clusters.delete_many({})
collection_1 = db['features']
collection_2 = db['clusters']
print "Number of docs removed in collection 'features' = {}".format(result_1.deleted_count)
print "Number of docs removed in collection 'clusters' = {}".format(result_2.deleted_count)

df = pd.read_csv('../FinalCapstoneData/ut-zap50k-data/meta-data-bin.csv')
dataset = Dataset(df)
dataset.subset_to_cols(['CID', 'Category'])
sample_df = dataset.subset_df.sample(NUM_SAMPLES)
image_paths = map(lambda x : ''.join(x.replace('-', '.'), '.jpg'), sample_df['CID'].values)
print "Total number of images = {}".format(len(image_paths))

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
output_layer = net['fc8']
model = cPickle.load(open(PICKLED_MODEL_FOLDER))
lasagne.layers.set_all_param_values(output_layer, model['values'])

# Theano function for the neural net output. The point is it is compiled only once.
x = T.tensor4('x')
y = lasagne.layers.get_output(net['fc7'], x, deterministic=True)
f = function([x], y)

p = Preprocessing()
failure_list = []

im_1 = p.preprocess('../FinalCapstoneData/flat_data/' + image_paths[0])
im_1 = floatX(im_1[np.newaxis])
X = f(im_1)
X_0 = X[0]
image_labels = [(0, image_paths[0])]

for image_path in image_paths[:]:
    try:
        im = p.preprocess('../FinalCapstoneData/flat_data/' + image_path)
        im = floatX(im[np.newaxis])
        features = f(im)
        collection_1.insert_one(create_doc(image_path, features))
        X_0 = np.vstack((X_0, features[0]))
        image_labels.append((count, image_path))
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        failure_list.append(image_path)

# Now cluster them using minibatck KMeans
model = MiniBatchKMeans(NUM_CLUSTERS)
cluster_assign = model.fit_predict(X_0)
cluster_centroids = model.cluster_centers_

# Build the mongo document (cluster info) to be inserted into the database
dic = defaultdict(list)
for index, cluster_id in enumerate(cluster_assign):
    dic[cluster_id].append(image_labels[index][1])

for cluster_id in dic:
    doc = {'cluster_id': str(cluster_id), 'images_in_cluster': dic[cluster_id], 'sparse_center' : sparcify([cluster_centroids[cluster_id]]) }
    collection_2.insert_one(doc)
