from flask import Flask, render_template, jsonify
import os
from flask import Flask, request, redirect, url_for, session
from werkzeug import secure_filename
import cPickle
import sys
sys.path.append("..")
import time
import building_dataset
import pandas as pd
from pymongo import MongoClient
import my_util
import ConvNeuralNet

cnn = ConvNeuralNet.ConvNeuralNet()
cnn.load()

client = MongoClient()
db = client['image_features']
collection = db['features']
cursor = collection.find({})
feature_df = pd.DataFrame(list(cursor))
image_ids = feature_df['image_id'].values

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
            top = top_rank_relevant_images(util.sparse(feat_vector))
            ids = zip(*top)[1]
            return redirect(url_for('my_carousel', image_ids=','.join(ids)))
    return render_template('index.html')

def top_rank_relevant_images(sparse_feat_vector):
      ranks = feature_df['sparse_features'].apply(lambda x : util.inner_prod(x, sparse_feat_vector)).values
      top_relevant_images = sorted(zip(ranks, image_ids), key=lambda x: x[0], reverse=True)

      return top_relevant_images[:5]

app.run(debug=True)
