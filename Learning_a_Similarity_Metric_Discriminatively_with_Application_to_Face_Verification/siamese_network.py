import csv
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from tqdm import tqdm

from glob_params import logger

TINY_IMAGENET_200_TEST = os.path.join('..', 'dataset', 'tiny-imagenet-200_test')
TRIPLET_NAME = 'triplet.csv'
ALL_IMG_NAME = 'all_img_name.csv'
CKPT = os.path.join('.', 'siamese.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')
epoch = 15


def generate_triplet(dataset_path: str):
    with open(TRIPLET_NAME, mode='w', encoding='utf8') as f:
        classes = glob(os.path.join(dataset_path, '*'))
        for class_1 in tqdm(classes):
            for class_2 in classes:
                class_1_images = glob(os.path.join(class_1, 'images', '*.JPEG'))
                class_2_images = glob(os.path.join(class_2, 'images', '*.JPEG'))
                if class_1 == class_2:
                    label = 0
                else:
                    label = 1
                for class_1_image in class_1_images:
                    for class_2_image in class_2_images:
                        f.writelines(class_1_image + ',' + class_2_image + ',' + str(label) + '\n')


def nets(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            net = slim.conv2d(inputs, 20, [5, 5], padding='VALID', scope='conv1')
            # net = tf.nn.relu(net)
            # net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool1')
            net = slim.conv2d(net, 50, [5, 5], padding='VALID', scope='conv2')
            # net = tf.nn.relu(net)
            # net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 500, scope='fc1')
            # net = slim.batch_norm(net)
            net = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
            net = tf.reshape(net, shape=[-1, 2])
            return net


def loss_func(X1, X2, Y, Q=100):
    dist = tf.reduce_sum(tf.abs(X1 - X2))
    L_g = 2 / Q * tf.square(dist)
    L_i = 2 * Q * tf.exp(-2.77 / Q * dist)
    loss = (1 - Y) * L_g + Y * L_i
    return loss, dist


def generate_all_filename(dataset_path: str):
    with open(ALL_IMG_NAME, mode='w', encoding='utf8') as f:
        classes = glob(os.path.join(dataset_path, '*'))
        for class_ in tqdm(classes):
            r = np.random.rand()
            g = np.random.rand()
            b = np.random.rand()
            a = np.random.rand()
            class_images = glob(os.path.join(class_, 'images', '*.JPEG'))
            for class_image in class_images:
                f.writelines(class_image + ',' + str(r) + ',' + str(g) + ',' + str(b) + ',' + str(a) + '\n')


def main():
    if not os.path.exists(TRIPLET_NAME):
        generate_triplet(os.path.join(TINY_IMAGENET_200_TEST, 'train'))
        generate_all_filename(os.path.join(TINY_IMAGENET_200_TEST, 'train'))

    image_1 = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])
    image_2 = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])
    Y = tf.placeholder(dtype=tf.float32)

    with tf.variable_scope('re1_re2') as scope:
        image_1_re = nets(image_1)
        scope.reuse_variables()
        image_2_re = nets(image_2)

    loss, dist = loss_func(X1=image_1_re, X2=image_2_re, Y=Y)

    opt = tf.train.AdamOptimizer().minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        if not os.path.exists(CKPT_VALID):
            for i in range(epoch):
                epoch_loss = 0
                csv_reader = csv.reader(open(TRIPLET_NAME, mode='r', encoding='utf8'))
                for line in tqdm(csv_reader):
                    img1_path = line[0]
                    img2_path = line[1]
                    if img1_path == img2_path:
                        continue
                    img1 = np.reshape(cv2.resize(cv2.imread(img1_path), dsize=(224, 224)),
                                      newshape=[1, 224, 224, 3]) / 127.5 - 1
                    img2 = np.reshape(cv2.resize(cv2.imread(img2_path), dsize=(224, 224)),
                                      newshape=[1, 224, 224, 3]) / 127.5 - 1
                    _, cost, dist11, re1, re2 = sess.run([opt, loss, dist, image_1_re, image_2_re],
                                                         feed_dict={image_1: img1, image_2: img2, Y: line[2]})
                    epoch_loss += cost
                logger.info('Epoch {} loss: {}'.format(i, epoch_loss))
                saver.save(sess, CKPT)
        else:
            saver.restore(sess, CKPT)
            csv_reader = csv.reader(open(ALL_IMG_NAME, mode='r', encoding='utf8'))
            for line in csv_reader:
                img = np.reshape(cv2.resize(cv2.imread(line[0]), dsize=(224, 224)),
                                 newshape=[1, 224, 224, 3]) / 127.5 - 1
                result = sess.run(image_1_re, feed_dict={image_1: img})
                plt.scatter([result[0][0]], [result[0][1]], c=[[float(line[1]), float(line[2]), float(line[3]), float(line[4])],])
            plt.savefig('scatter.png')


if __name__ == '__main__':
    main()
