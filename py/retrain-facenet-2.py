# https://github.com/davidsandberg/facenet/issues/62#issuecomment-293444986

import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import importlib
import random
from PIL import Image
import glob
import facenet


class FaceNetPredictor:
    # convert images to their tensor representation
    def convert(self, f):
        current = Image.open(f)
        image_size = current.size[0]
        file_contents = tf.read_file(f)
        name = f.rsplit('/')[-2]
        image = tf.image.decode_png(file_contents)  # , channels=3)
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        image.set_shape((image_size, image_size, 3))
        #image = tf.image.per_image_whitening(image)
        image = tf.expand_dims(image, 0, name=name)
        return image

    # define logits
    def calc(self, image):
        model_def = "models.inception_resnet_v1"
        network = importlib.import_module(model_def)
        prelogits, _ = network.inference(images_placeholder, 1.0,
                                         phase_train=False, weight_decay=0.0, reuse=False)
        batch_norm_params = {
            # Decay for the moving averages
            'decay': 0.995,
            # epsilon to prevent 0s in variance
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            # Only update statistics during training mode
            'is_training': False
        }
        bottleneck = slim.fully_connected(prelogits, 128, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(
                                              stddev=0.1),
                                          weights_regularizer=slim.l2_regularizer(
                                              0.0),
                                          normalizer_fn=slim.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          scope='Bottleneck', reuse=False)
        logits = slim.fully_connected(bottleneck, 2, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(
                                          0.),
                                      scope='Logits', reuse=False)
        return logits

     # make prediction
    def predict(self, session, image, logits):
        predictions = tf.nn.softmax(logits)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        softmax = session.run(predictions, feed_dict={
                              images_placeholder: image.eval()})
        softmax_out = softmax[0].argmax()
        print("full vector: {}, max: {}", softmax, softmax_out)


# get file list
files = glob.glob('/tmp/image_samples/*')
random.shuffle(files)
model_dir = '/tmp/facenet_model/20170410-180000'

# load graph into session from checkpoint
with tf.Graph().as_default():
    with tf.Session() as sess:
        fp = FaceNetPredictor()
        print('Model directory: %s' % model_dir)
        meta_file, ckpt_file = facenet.get_model_filenames(
            os.path.expanduser(model_dir))

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        facenet.load_model(model_dir, meta_file, ckpt_file)

        images_placeholder = tf.placeholder(
            tf.float32, shape=(None, 182, 182, 3), name='input')
        Logits = fp.calc(fp.convert(files[0]))

        for x, f in enumerate(files):
            print("Predicting for image: %s" % f)
            image = fp.convert(f)
            fp.predict(sess, image, Logits)
