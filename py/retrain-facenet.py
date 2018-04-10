# https://github.com/davidsandberg/facenet/issues/62#issuecomment-265002615

# coding: utf-8
import tensorflow as tf
import os
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
import numpy as np
import importlib
import random
import Image
import glob

# facenet code for getting model files from dir


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt_files = [s for s in files if 'ckpt' in s]
    if len(ckpt_files) == 0:
        raise ValueError(
            'No checkpoint file found in the model directory (%s)' % model_dir)
    elif len(ckpt_files) == 1:
        ckpt_file = ckpt_files[0]
    else:
        ckpt_iter = [(s, int(s.split('-')[-1]))
                     for s in ckpt_files if 'ckpt' in s]
        sorted_iter = sorted(ckpt_iter, key=lambda tup: tup[1])
        ckpt_file = sorted_iter[-1][0]
    return meta_file, ckpt_file

# import model checkpoint


def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    print model_dir_exp
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(),
                  os.path.join(model_dir_exp, ckpt_file))
    return

# convert images to their tensor representation


def convert(f):
    current = Image.open(f)
    image_size = current.size[0]
    file_contents = tf.read_file(f)
    name = f.rsplit('/')[-2]
    image = tf.image.decode_png(file_contents)  # , channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
        image, image_size, image_size)
    image.set_shape((image_size, image_size, 3))
    image = tf.image.per_image_whitening(image)
    image = tf.expand_dims(image, 0, name=name)
    return image

# define logits


def calc(image):
    with graph.as_default():
        network = importlib.import_module(
            'src.models.inception_resnet_v1', 'inference')
        prelogits, _ = network.inference(images_placeholder, 1.0,
                                         phase_train=False, weight_decay=0.0, reuse=False)
        logits = slim.fully_connected(prelogits, 7, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(
                                          0.),
                                      scope='Logits', reuse=False)
        return logits

 # make prediction


def predict(image, logits):
    predictions = tf.nn.softmax(logits)
    session.run(tf.initialize_all_variables())
    softmax = session.run(predictions, feed_dict={
                          images_placeholder: image.eval()})
    softmax_out = softmax[0].argmax()

    # get target label from image tensor
    target = int(image.name[:1])

    # construct output dict for analysis
    outDict = {'target': target, 'prediction': softmax_out,
               'truth': target == softmax_out}
    with open('test_results.txt', 'a') as myfile:
        myfile.write(str(outDict) + ', ')
    print outDict


# get file list
files = glob.glob(os.path.join(os.getcwd(), 'fer2013'))
random.shuffle(files)
model_dir = os.path.join(os.getcwd(), 'facenet_modeuls/20170512-110547')

# load graph into session from checkpoint
meta_file, ckpt_file = get_model_filenames(os.path.expanduser(model_dir))
session = tf.InteractiveSession()
load_model(model_dir, meta_file, ckpt_file)
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()

images_placeholder = tf.placeholder(
    tf.float32, shape=(None, 160, 160, 3), name='input')
Logits = calc(convert(files[0]))

for x, f in enumerate(files):
    image = convert(f)
    predict(image, Logits)
