import logging
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from glob_params import CIFAR10_DATASET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_data():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    return unpickle(os.path.join(CIFAR10_DATASET, 'data_batch_1'))


def net(input):
    with tf.variable_scope('ConvNet') as scope:
        with slim.arg_scope([slim.layers.conv2d], padding='SAME', normalizer_fn=slim.batch_norm):
            conv1 = slim.layers.conv2d(inputs=input, num_outputs=96, kernel_size=7, stride=2, scope='conv1')


def train():
    pass


if __name__ == '__main__':
    print(read_data()['data'].shape)
