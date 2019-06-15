import os

import numpy as np
import tensorflow as tf

from glob_params import logger

CKPT = os.path.join('.', 'visual.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')
checkpoint_file = tf.train.latest_checkpoint('.')

conv3_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 64, 64], name='deconv3_w')
conv2_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 32, 64], name='deconv2_w')
conv1_w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 1, 32], name='deconv1_w')


def deconvnet_conv1(input):
    output = deconv2d(input, conv1_w, [1, 28, 28, 1], [1, 2, 2, 1], 'SAME', 'deconv1')

    return output


def deconvnet_conv2(input):
    deconv2 = deconv2d(input, conv2_w, [1, 14, 14, 32], [1, 2, 2, 1], 'SAME', 'deconv2')
    output = deconv2d(deconv2, conv1_w, [1, 28, 28, 1], [1, 2, 2, 1], 'SAME', 'deconv1')

    return output


def deconvnet_conv3(input):
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


def run(conv1, conv2, conv3):
    x = tf.reshape(conv1, shape=[1, 14, 14, 32])
    outputx = deconvnet_conv1(x)
    y = tf.reshape(conv2, shape=[1, 7, 7, 64])
    outputy = deconvnet_conv2(y)
    z = tf.reshape(conv3, shape=[1, 4, 4, 64])
    outputz = deconvnet_conv3(z)

    with tf.Session() as sess:
        from tensorflow.python import pywrap_tensorflow
        ckpt = tf.train.get_checkpoint_state('./')
        ckpt_path = ckpt.model_checkpoint_path

        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        param_dict = reader.get_variable_to_shape_map()
        logger.info('Params: {}'.format(param_dict))
        conv3_w_ = reader.get_tensor('ConvNet/conv3/conv_w')
        conv2_w_ = reader.get_tensor('ConvNet/conv2/conv_w')
        conv1_w_ = reader.get_tensor('ConvNet/conv1/conv_w')

        image1 = sess.run(outputx, feed_dict={conv1_w: conv1_w_})

        image2 = sess.run(outputy, feed_dict={conv2_w: conv2_w_,
                                              conv1_w: conv1_w_})

        image3 = sess.run(outputz, feed_dict={conv3_w: conv3_w_,
                                              conv2_w: conv2_w_,
                                              conv1_w: conv1_w_})
        return np.array(image1).reshape(28, 28, 1) * 255, \
               np.array(image2).reshape(28, 28, 1) * 255, \
               np.array(image3).reshape(28, 28, 1) * 255
