import numpy as np
from building_dataset import Preprocessing
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
import cPickle
import time

# The VGG convolutional neural net architecture and pretrained weights
class ConvNeuralNet(object):
    def __init__(self, pickle_path, image_upload_folder):
        self.neural_net = {}
        self.output_layer = None
        self.CLASSES = None
        self.pickle_path = pickle_path
        self.image_upload_folder = image_upload_folder

    def load(self):
        vgg_model = cPickle.load(open(self.pickle_path, 'rb'))
        self.CLASSES = vgg_model['synset words']
        self.neural_net['input'] = InputLayer((None, 3, 224, 224))
        self.neural_net['conv1'] = ConvLayer(self.neural_net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
        self.neural_net['norm1'] = NormLayer(self.neural_net['conv1'], alpha=0.0001)
        self.neural_net['pool1'] = PoolLayer(self.neural_net['norm1'], pool_size=3, stride=3, ignore_border=False)
        self.neural_net['conv2'] = ConvLayer(self.neural_net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
        self.neural_net['pool2'] = PoolLayer(self.neural_net['conv2'], pool_size=2, stride=2, ignore_border=False)
        self.neural_net['conv3'] = ConvLayer(self.neural_net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.neural_net['conv4'] = ConvLayer(self.neural_net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.neural_net['conv5'] = ConvLayer(self.neural_net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.neural_net['pool5'] = PoolLayer(self.neural_net['conv5'], pool_size=3, stride=3, ignore_border=False)
        self.neural_net['fc6'] = DenseLayer(self.neural_net['pool5'], num_units=4096)
        self.neural_net['drop6'] = DropoutLayer(self.neural_net['fc6'], p=0.5)
        self.neural_net['fc7'] = DenseLayer(self.neural_net['drop6'], num_units=4096)
        self.neural_net['drop7'] = DropoutLayer(self.neural_net['fc7'], p=0.5)
        self.neural_net['fc8'] = DenseLayer(self.neural_net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
        self.output_layer = self.neural_net['fc8']
        lasagne.layers.set_all_param_values(self.output_layer, vgg_model['values'])

    def featurize(self, image):
        """ INPUT: input image
            OUTPUT: 4096 dimensional feature vector
        """
        p = Preprocessing()
        image = p.preprocess(''.join(self.image_upload_folder, image))
        image = floatX(image[np.newaxis])
        features = np.array(lasagne.layers.get_output(self.neural_net['fc7'], image, deterministic=True).eval())

        return features

    def get_top5_proba(self, image):
        p = Preprocessing()
        image = p.preprocess(image)
        image = floatX(image[np.newaxis])
        prob = np.array(lasagne.layers.get_output(self.output_layer, image, deterministic=True).eval())
        top5 = np.argsort(prob[0])[-1:-6:-1]

        return top5

    def print_top5_preds(self, image):
        for no, label in enumerate(self.get_top5_proba(image)):
            print "{}.{}".format(no, self.CLASSES[label])

    def get_feature_layer(self):
        return  self.net['fc7']
