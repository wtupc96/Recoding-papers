import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.contrib.slim import nets

from glob_params import CIFAR10_DATASET, RESNET_MODEL, logger


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def predict(is_training, num_classes, preprocessed_inputs):
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        net, endpoints = nets.resnet_v2.resnet_v2_152(preprocessed_inputs, num_classes, is_training)

    with tf.Session() as sess:
        init_fn = slim.assign_from_checkpoint_fn(model_path=RESNET_MODEL,
                                                 var_list=slim.get_variables_to_restore(), ignore_missing_vars=True)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        init_fn(session=sess)
        result = sess.run(net)
        if not num_classes:
            result = np.reshape(result, newshape=[-1, 2048])
    return result


if __name__ == '__main__':
    data_filenames = glob(os.path.join(CIFAR10_DATASET, 'data_batch_*'))
    images_train = np.zeros(shape=[100, 224, 224, 3])
    labels_train = np.zeros(shape=[images_train.shape[0], 1])
    for df in data_filenames:
        data = unpickle(df)
        images = data[b'data']
        labels = data[b'labels']
        images = np.array(images).reshape([10000, 32, 32, 3])
        for cnt in range(len(images)):
            if cnt == images_train.shape[0]:
                break
            idx = np.random.randint(low=0, high=10000)
            image = cv2.resize(images[idx], dsize=(224, 224))
            images_train[cnt] = image
            labels_train[cnt] = labels[idx]
        break
    images_train = images_train.astype(np.float32)
    logger.info('Predict using RESNET_V2_152...')
    re = predict(False, None, images_train)

    x_train, x_test, y_train, y_test = train_test_split(re, labels_train, random_state=1, train_size=0.9)

    clf = svm.SVC(C=0.75, kernel='rbf', gamma=5, decision_function_shape='ovr', max_iter=-1)
    clf.fit(x_train, y_train.ravel())
    sc_train = clf.score(x_train, y_train)
    logger.info('Training score: {}'.format(sc_train))

    sc_test = clf.score(x_test, y_test)
    logger.info('Testing score: {}'.format(sc_test))
