import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import LinearSVC
import cv2
import os
import time


class Dataset(object):

    def __init__(self, data_frame):
        self.df = data_frame
        self.col_full_df = data_frame.columns
        self.subset_df = None

    def _is_kept(x, col_names):
        for name in col_names:
            if x.startswith(name):
                return True
        return False

    def _get_int_class_labels(self):
        keys = map(lambda x : ''.join(map(str, x)), self.subset_df.values[:, 1:])
        int_class_labels = []
        dic = {}
        count = 0
        for key in keys:
            if key not in dic:
                dic[key] = count
                int_class_labels.append(count)
                count += 1
            else:
                int_class_labels.append(dic[key])

    def subset_to_cols(self, col_names, class_label=True):
        self.subset_df = self.df[filter(lambda x : _is_kept(x, col_names), self.col_full_df)]
        self._get_int_class_labels()
        self.subset_df['class'] = pd.Series(np.array(int_class_labels), index=self.subset_df.index)

    def get_class_by_image_id(self, image_id):
        return self.subset_df[self.subset_df['CID']==image_id]['class'].values[0]


class Preprocessing(object):

    def __init__(self, resize_to=256, half_size=112):
        self.resize_val = resize_to
        self.half_size=half_size
        self.descriptor_list = []

    def preprocess(self, full_image_path):
        im = cv2.imread(full_image_path)
        height, width, _  = im.shape
        if height < width:
            im = cv2.resize(im, (self.resize_val, width*self.resize_val/height))
        else:
            im = cv2.resize(im, (height*self.resize_val/width, self.resize_val))
        # Crop the image
        height, width, _ = im.shape
        im = im[height/2 - self.half_size : height/2 + self.half_size, width/2 - self.half_size : width/2 + self.half_size]
        # Convert axes
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        # Convert to BGR
        im = im[::-1, :, :]
        # Subtract the median and filter noise
        im = im - np.median(im)
        im = cv2.GaussianBlur(im, (3, 3), 0)

        return im

    def get_SIFT_descriptor(self, full_image_path):
        im = cv2.imread(full_image_path)

        feature_detector = cv2.FeatureDetector_create("SIFT")
        descriptor_extractor = cv2.DescriptorExtractor_create("SIFT")
        key_points = feature_detector.detect(im)
        key_points, descriptor = descriptor_extractor.compute(im, key_points)

        return descriptor

    def get_SIFT_decriptor_batch(self, image_folder_path, image_name_list):
            self.descriptor_list = []
            for image_name in image_name_list:
                im = cv2.imread('/'.join(image_folder_path, image_name))

                feature_detector = cv2.FeatureDetector_create("SIFT")
                descriptor_extractor = cv2.DescriptorExtractor_create("SIFT")
                key_points = feature_detector.detect(im)
                key_points, descriptor = descriptor_extractor.compute(im, key_points)
                self.descriptor_list.append((image_name, descriptor))

            descriptors = self.descriptor_list[0][1]
            for _, descriptor in self.descriptor_list[1:]:
                descriptors = np.vstack((descriptors, descriptor))

            return descriptors

    def calc_visual_term_frequency(self, descriptors, image_paths, num_clusters=100):
        visual_term_frequency = np.zeros((len(image_paths), num_clusters), dtype='float32')
        visual_word_count = np.zeros((len(image_paths), num_clusters), dtype='float32')
        model = KMeans(n_clusters=num_clusters, n_jobs=-1)
        model.fit(descriptors)
        for i in xrange(len(image_paths)):
            words = model.predict(self.descriptor_list[i][1])
            for word in words:
                visual_term_frequency[i][word] += 1./len(words)
                visual_word_count[i][word] += 1

        num_occurences = np.sum((visual_word_count > 0)*1, axis=0)
        idf = np.array(np.log((len(image_paths) + 1)/(1.*num_occurences + 1)), dtype='float32')
        tfidf = visual_term_frequency*idf

        return tfidf
