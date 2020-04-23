import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imp
import innvestigate
import innvestigate.applications
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import requests
import urllib.request

print("Loading model")

eutils = imp.load_source(
    "utils", "./utils.py")
imgnetutils = imp.load_source(
    "utils_imagenet", "./utils_imagenet.py")

# Load the model definition.
tmp = getattr(innvestigate.applications.imagenet,
              os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns="relu")


global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('compareMethods.h5')
global graph
graph = tf.get_default_graph()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        print('here in post')
        data = request.get_data()
        print('here')
        imageURL = data.decode("utf-8")
        # Add photo
        print("imageURL", str(imageURL).replace('"', ""))
        Picture_request = requests.get(str(imageURL).replace('"', ""))
        if Picture_request.status_code == 200:
            with open("utils/images/n02799071_986.jpg", 'wb') as f:
                f.write(Picture_request.content)

        predictions = "predictions"

    # Step 3
    with graph.as_default():
        print('graph')
        set_session(sess)
        input_range = net["input_range"]
        noise_scale = (input_range[1]-input_range[0]) * 0.1
        images, label_to_class_name = eutils.get_imagenet_data(
            net["image_shape"][0])
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
        text = []
        # Step 4
        for i, (x, y) in enumerate(images):
            x = x[None, :, :, :]
            x_pp = imgnetutils.preprocess(x, net)
            # Predict final activations, probabilites, and label.
            # presm = model_wo_softmax.predict_on_batch(x_pp)[0]
            prob = model.predict_on_batch(x_pp)[0]
            y_hat = prob.argmax()
            # Save prediction info:
            text.append((
                # "%.2f" % presm.max(),             # pre-softmax logits
                "%.2f" % prob.max(),              # probabilistic softmax output
                "%s" % label_to_class_name[y_hat]  # predicted label
            ))

        predictions = {
            # "class1": number_to_class[index[9]],
            # "class2": number_to_class[index[8]],
            # "class3": number_to_class[index[7]],
            # "prob1": probabilities[index[9]],
            # "prob2": probabilities[index[8]],
            # "prob3": probabilities[index[7]],
        }
        print('text', text[0])
        predictions = text

    # Step 5
    return render_template('predict.html', predictions=predictions)


app.run(host='0.0.0.0', port=80)
