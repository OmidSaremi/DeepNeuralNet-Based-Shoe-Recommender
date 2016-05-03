import cv2
import numpy as np

# This script overlays SIFT keypoints detected in an input image on the image.
# This script was used to generate an example for the presentation.
path_to_example = '../FinalCapstoneData/flat_data/7447079.264230.jpg'
img = cv2.imread(path_to_example)
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

feature_detector = cv2.FeatureDetector_create("SIFT")
descriptor_extractor = cv2.DescriptorExtractor_create("SIFT")

key_points = feature_detector.detect(gray_scale)
key_points, descriptor = descriptor_extractor.compute(gray_scale, key_points)

print "Shape of the SIFT descriptor = ", descriptor.shape
img=cv2.drawKeypoints(gray_scale,key_points)
cv2.imwrite('sift_keypoints_on_image.jpg', img)
