import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

print(tf.VERSION)

meta_path = './facenet2/model.meta'

checkpoint_folder_path = './facenet1/'
checkpoint_path = './facenet2/model.ckpt-500000'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, checkpoint_path)

    tensor_names = chkp.print_tensors_in_checkpoint_file(
        checkpoint_path, tensor_name='', all_tensors=True)

    print(tensor_names)
