import os
import cPickle
import numpy as np
import re
from building_dataset import *
from ConvNeuralNet import ConvNeuralNet
import pandas as pd
'''
from __future__ import print_function
'''
import itertools
import pickle
import sys
import lasagne
import theano
from lasagne.layers import dnn
import theano.tensor as T
import time
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.regularization import l2, l1
from lasagne.utils import floatX

half_size = 25
BATCH_SIZE = 90
LEARNING_RATE = 0.00001
MOMENTUM = 0.9

df = pd.read_csv('../FinalCapstoneData/ut-zap50k-data/meta-data-bin.csv')
dataset = Dataset(df)
dataset.subset_to_cols(['CID', 'Category'])

print dataset.subset_df.shape
print dataset.subset_df.groupby('class').count()

c0 = map(lambda x : x.replace('-', '.' ) + '.jpg', dataset.subset_df[dataset.subset_df['class']==0]['CID'].values)
c1 = map(lambda x : x.replace('-', '.' ) + '.jpg', dataset.subset_df[dataset.subset_df['class']==1]['CID'].values)
c2 = map(lambda x : x.replace('-', '.' ) + '.jpg', dataset.subset_df[dataset.subset_df['class']==2]['CID'].values)
c3 = map(lambda x : x.replace('-', '.' ) + '.jpg', dataset.subset_df[dataset.subset_df['class']==3]['CID'].values)

np.random.seed(1233)
n_per_cls = 400    #400
s_c0 = np.random.choice(c0, n_per_cls, replace=False)
s_c1 = np.random.choice(c1, n_per_cls, replace=False)
s_c2 = np.random.choice(c2, n_per_cls, replace=False)
s_c3 = np.random.choice(c3, n_per_cls, replace=False)
print map(len, [s_c0, s_c1, s_c2, s_c3])

siamese_dataset = zip(s_c0, [0 for i in range(n_per_cls)]) + zip(s_c1, [1 for i in range(n_per_cls)])+ \
zip(s_c2, [2 for i in range(n_per_cls)])+ \
zip(s_c3, [3 for i in range(n_per_cls)])

np.random.shuffle(siamese_dataset)
print "First 100 in the shuffled version are: ", siamese_dataset[:100]

siamese_dataset_train = siamese_dataset[: 4 * n_per_cls * 7/10]
siamese_dataset_valid = siamese_dataset[4 * n_per_cls * 3/10 :]

print "perc sim train", np.mean(zip(*siamese_dataset_train)[1])
print "perc sim valid", np.mean(zip(*siamese_dataset_valid)[1])

def load_data(siamese_dataset, siamese_dataset_valid):
    p = Preprocessing(resize_to=70, half_size=25)
    im1 = p.preprocess('../FinalCapstoneData/flat_data/' + siamese_dataset[0][0])
    im1 = floatX(im1[np.newaxis])
    X_train = im1.reshape(1, 3, 2*half_size, 2*half_size)
    y_train = [siamese_dataset[0][1]]
    c = 0
    for image in siamese_dataset[1:]:
            im = p.preprocess('../FinalCapstoneData/flat_data/' + image[0])
            im = floatX(im[np.newaxis])
            X_train = np.concatenate((X_train, im.reshape(1, 3, 2*half_size, 2*half_size)), axis=0)
            y_train.append(image[1])
            c+=1
            print c
    print "x :", X_train.shape
    print np.array(y_train).size
    y_train = np.array(y_train)

    p_val = Preprocessing(resize_to=70, half_size=25)
    im1_val = p_val.preprocess('../FinalCapstoneData/flat_data/' + siamese_dataset_valid[0][0])
    im1_val = floatX(im1_val[np.newaxis])
    X_valid = im1_val.reshape(1, 3, 2*half_size, 2*half_size)
    y_valid = [siamese_dataset_valid[0][1]]
    c = 0
    for image_val in siamese_dataset_valid[1:]:
            im_val = p_val.preprocess('../FinalCapstoneData/flat_data/' + image_val[0])
            im_val = floatX(im_val[np.newaxis])
            X_valid = np.concatenate((X_valid, im_val.reshape(1, 3, 2*half_size, 2*half_size)), axis=0)
            y_valid.append(image_val[1])
            c+=1
            print c
    print "x :", X_valid.shape
    print np.array(y_valid).size
    y_valid = np.array(y_valid)

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        y_train=y_train.astype('int32'),
        X_valid=lasagne.utils.floatX(X_valid),
        y_valid=y_valid.astype('int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        )

def build_model(input_width, input_height, batch_size=BATCH_SIZE):
    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, input_width, input_height),
    )

    l_cv_1 = ConvLayer(l_in, num_filters=32, filter_size=2)

    l_pool_1 = PoolLayer(l_cv_1, pool_size=2)
    l_drop_1 = DropoutLayer(l_pool_1, p=0.5)
    l_cv_2 = ConvLayer(l_pool_1, num_filters=64, filter_size=2)
    l_pool_2 = PoolLayer(l_cv_2, pool_size=2)
    l_drop_2 = DropoutLayer(l_pool_2, p=0.5)
    l_h_1 =DenseLayer(l_drop_2, num_units=200)
    l_drop_3 =  DropoutLayer(l_h_1, p=0.5)
    l_h_2 = DenseLayer(l_drop_3, num_units=200)
    l_out = DenseLayer(l_h_2, num_units=4, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def create_iter_functions(output_layer, learning_rate=LEARNING_RATE,momentum=MOMENTUM):
    X = T.tensor4()
    y = T.ivector()
    a = lasagne.layers.get_output(output_layer, X, deterministic=True)

    loss = lasagne.objectives.categorical_crossentropy(a, y)
    loss = loss.mean()
    all_params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate, momentum)
    iter_train = theano.function([X, y], loss, updates=updates)
    iter_valid = theano.function([X, y], loss)

    return dict(train=iter_train, valid=iter_valid)

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    batch_train_losses = []
    for batch_index in range(num_batches_train):
        batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
        batch_train_loss = iter_funcs['train'](dataset['X_train'][batch_slice], dataset['y_train'][batch_slice])
        batch_train_losses.append(batch_train_loss)
    avg_train_loss = np.mean(batch_train_losses)

    batch_valid_losses = []
    for batch_index in range(num_batches_valid):
        batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
        batch_valid_loss = iter_funcs['valid'](dataset['X_valid'][batch_slice], dataset['y_valid'][batch_slice])
        batch_valid_losses.append(batch_valid_loss)
    avg_valid_loss = np.mean(batch_valid_losses)

    return {'train_loss': avg_train_loss, 'valid_loss': avg_valid_loss}

def main(num_epochs):
    print("Loading data...")
    dataset = load_data(siamese_dataset_train, siamese_dataset_valid)

    print("Building model and compiling functions...")
    output_layer = build_model(2 * half_size, 2 * half_size)

    iter_funcs = create_iter_functions(output_layer)

    print("Starting training...")
    now = time.time()
    try:
        i = 0
        while 1:
            epoch = train(iter_funcs, dataset)
            i += 1
            epoch['number'] = i
            print("Epoch {} of {} took {:.3f}s".format(epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    return output_layer

dataset = load_data(siamese_dataset_train, siamese_dataset_valid)
l_out = build_model(2*half_size, 2*half_size)
iter_funcs = create_iter_functions(l_out)
l_out = main(1000)
