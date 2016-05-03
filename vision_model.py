from building_dataset import *
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import time

# Using hand-crafted features (SIFT) for image classification

NUM_SAMPLES = 500
VISUAL_VOCAB = 200

t_prep = time.time()
df = pd.read_csv('../FinalCapstoneData/ut-zap50k-data/meta-data-bin.csv')
dataset = Dataset(df)
dataset.subset_to_cols(['CID', 'Category'])
sample_df = dataset.subset_df.sample(NUM_SAMPLES)
print sample_df.groupby('class').count()
image_name_list = map(lambda x: x.replace('-', '.') + '.jpg', sample_df['CID'].values)
y_label = sample_df['class'].values
print "Prep time is {}".format(time.time() - t_prep)

p = Preprocessing()
t_des = time.time()
descriptors = p.get_SIFT_decriptor_batch('../FinalCapstoneData/flat_data', image_name_list)
print "Calculating SIFT descriptors took {}".format(time.time() - t_des)
t_tfidf = time.time()
tfidf = p.calc_visual_term_frequency(descriptors, image_name_list, VISUAL_VOCAB)
print "Vectorizing took {}".format(time.time() - t_tfidf)

model = SVC(kernel='rbf', gamma=5, class_weight='auto')
model.fit(tfidf, y_label)
print "SVC score: ", model.score(tfidf, y_label)
y_pred = model.predict(tfidf)
print  Counter(y_label[y_pred==y_label])
