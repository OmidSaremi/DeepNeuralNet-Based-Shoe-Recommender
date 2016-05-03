from flask import Flask, render_template, jsonify
from flask import Flask, request, redirect, url_for, session
from werkzeug import secure_filename
from pymongo import MongoClient
import pandas as pd
import building_dataset
import util
import ConvNeuralNet
import os
import cPickle
import sys
import time

sys.path.append("..")
cnn = ConvNeuralNet.ConvNeuralNet()
cnn.load()

client = MongoClient()
db = client['image_features']
collection = db['features']
collection_2 = db['clusters']

cursor_2 = collection_2.find({})
clusters_df = pd.DataFrame(list(cursor_2))

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/carousel/<image_ids>')
def my_carousel(image_ids):
    return render_template('carousel.html', image_ids=image_ids)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            feat_vector = cnn.featurize(filename)
            top = top_rank_relevant_images(my_util.sparse(feat_vector))
            ids = zip(*top)[1]
            return redirect(url_for('my_carousel', image_ids=','.join(ids)))
    return render_template('index.html')

def top_rank_relevant_images(sparse_feat_vector):
      lst = _find_cluster(sparse_feat_vector)
      subset_feature_df = pd.DataFrame(lst)
      ranks = subset_feature_df['sparse_features'].apply(lambda x : util.inner_prod(x, sparse_feat_vector)).values
      top_relevant_images = sorted(zip(ranks, subset_feature_df['image_id'].values), key=lambda x: x[0], reverse=True)

      return top_relevant_images[:5]

def _find_cluster(sparse_feat_vector):
    cluster_ranks = clusters_df['sparse_center'].apply(lambda x : util.inner_prod(x, sparse_feat_vector)).values
    rank_sorted_clusters = sorted(zip(cluster_ranks, clusters_df['cluster_id'].values), key=lambda x : x[0], reverse=True)
    top_cluster_num = rank_sorted_clusters[0][1]
    # Query the database
    docs = list(collection_2.find({'cluster_id': top_cluster_num}))
    # List of images in cluster
    image_list = docs[0]['images_in_cluster']
    # Query features collection on the image list
    return list(collection.find({'image_id': {'$in': image_list}}))

app.run(debug=True)
