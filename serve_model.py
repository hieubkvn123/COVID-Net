import numpy as np
import tensorflow.compat.v1 as tf
import os, argparse
import cv2

from data import (
    process_image_file,
    process_image_file_medusa,
    process_image,
    process_image_medusa
)

from flask import Flask

app = Flask(__name__)

# Config
weightspath = 'models/COVIDNet-CXR-2'
metaname = 'model.meta'
ckptname = 'model'
n_classes = 2
imagepath = 'assets/ex-covid.jpeg'
in_tensorname = 'input_1:0'
out_tensorname = 'norm_dense_2/Softmax:0'
top_percent = 0.08
input_size = 480

# To remove TF Warnings
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For COVID-19 positive/negative detection
mapping = {'negative': 0, 'positive': 1}
inv_mapping = {0: 'negative', 1: 'positive'}
mapping_keys = list(mapping.keys())

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
sess = tf.Session(config=config)
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
saver.restore(sess, os.path.join(weightspath, ckptname))

graph = tf.get_default_graph()

@app.route("/", methods=["GET"])
def home():
    return 'API is running'

@app.route("/predict_single_image", methods=["POST"])
def predict_single_image():
    image_tensor = graph.get_tensor_by_name(in_tensorname)
    pred_tensor = graph.get_tensor_by_name(out_tensorname)

    x = process_image_file(imagepath, input_size, top_percent=top_percent)
    x = x.astype('float32') / 255.0
    feed_dict = {image_tensor: np.expand_dims(x, axis=0)}

    pred = sess.run(pred_tensor, feed_dict=feed_dict)

    print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
    print('Confidence')
    print(' '.join('{}: {:.3f}'.format(cls.capitalize(), pred[0][i]) for cls, i in mapping.items()))
    print('**DISCLAIMER**')
    print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')

    return {'_code' : 'success',
            '_label' : 1,
            '_confidence' : 0.83}

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=8889, debug=False)
