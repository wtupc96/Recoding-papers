import os

import tensorflow as tf

CKPT = os.path.join('.', 'visual.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')
checkpoint_file = tf.train.latest_checkpoint('.')

conv3_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 64, 64], name='deconv3_w')
conv2_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 32, 64], name='deconv2_w')
conv1_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 1, 32], name='deconv1_w')


def deconvnet(input):
    # with graph.as_default():
    #     saver = tf.train.import_meta_graph("{}.meta".format(CKPT))
    # with tf.Session() as sess:
    #     # Load the saved meta graph and restore variables
    #     saver.restore(sess, checkpoint_file)
    #     # w = graph.get_tensor_by_name('conv1')
    # conv1_w = graph.get_operation_by_name("ConvNet/conv1/conv_w").outputs[0]
    # conv2_w = graph.get_operation_by_name("ConvNet/conv2/conv_w").outputs[0]
    # conv3_w = graph.get_operation_by_name("ConvNet/conv3/conv_w").outputs[0]

    deconv3 = deconv2d(input, conv3_w, [1, 7, 7, 64], [1, 2, 2, 1], 'SAME', 'deconv3')
    deconv2 = deconv2d(deconv3, conv2_w, [1, 14, 14, 32], [1, 2, 2, 1], 'SAME', 'deconv2')
    output = deconv2d(deconv2, conv1_w, [1, 28, 28, 1], [1, 2, 2, 1], 'SAME', 'deconv1')

    return output


def deconv2d(input, filter, output_shape, strides, padding, name):
    with tf.variable_scope(name_or_scope=name):
        return tf.nn.conv2d_transpose(value=input,
                                      filter=filter,
                                      output_shape=output_shape,
                                      strides=strides,
                                      padding=padding)


def run(X):
    x = tf.reshape(X, shape=[1, 4, 4, 64])
    output = deconvnet(x)

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(CKPT))
        with tf.Session() as sess:
            # Load the saved meta graph and restore variables
            saver.restore(sess, checkpoint_file)
            # w = graph.get_tensor_by_name('conv1')
            conv1_w_ = graph.get_operation_by_name("ConvNet/conv1/conv_w")
            conv2_w_ = graph.get_operation_by_name("ConvNet/conv2/conv_w")
            conv3_w_ = graph.get_operation_by_name("ConvNet/conv3/conv_w")

    with tf.Session() as sess:
        image = sess.run(output, feed_dict={conv3_w: conv3_w_, conv2_w: conv2_w_, conv1_w: conv1_w_})
        import cv2
        import numpy as np
        cv2.imshow('image', np.array(image).reshape(28, 28, 1))
        cv2.waitKey()
