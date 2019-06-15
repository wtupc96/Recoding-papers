import sys

sys.path.append('/home/se-oypc/recoding_papers')

import os
import tensorflow as tf
from glob_params import MNIST_DATASET, logger
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np

batch_size = 1000
epoch = 100

mnist = read_data_sets(train_dir=MNIST_DATASET, one_hot=True)

CKPT = os.path.join('.', 'visual.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')


def read_data(file):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    return unpickle(file)


def net(input):
    with tf.variable_scope('ConvNet') as scope:
        conv1 = conv2d(input, 32, 3, 3, 2, 2, 'SAME', True, True, 'conv1')
        print(conv1.shape)
        conv2 = conv2d(conv1, 64, 3, 3, 2, 2, 'SAME', True, True, 'conv2')
        print(conv2.shape)
        conv3 = conv2d(conv2, 64, 3, 3, 2, 2, 'SAME', True, True, 'conv3')
        print(conv3.shape)
        conv_output = tf.layers.flatten(inputs=conv3, name='flatten')
        print(conv_output.shape)
        fc1 = tf.contrib.layers.fully_connected(inputs=conv_output, num_outputs=100,
                                                normalizer_fn=tf.layers.batch_normalization)
        print(fc1.shape)
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=50,
                                                normalizer_fn=tf.layers.batch_normalization)
        print(fc2.shape)

        output = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=10, activation_fn=None)
        print(output.shape)
        return conv1, conv2, conv3, output


def conv2d(input, output_ch, f_h, f_w, s_h, s_w, padding, do_norm, do_relu, name):
    with tf.variable_scope(name_or_scope=name):
        input_ch = int(input.get_shape()[-1])
        w = tf.get_variable(name='conv_w', shape=[f_h, f_w, input_ch, output_ch],
                            initializer=tf.truncated_normal_initializer(stddev=0.05))
        output = tf.nn.conv2d(input=input,
                              filter=w,
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name='conv')
        b = tf.get_variable(name='conv_b', shape=[output_ch], initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(output, bias=b)
        if do_norm:
            # output = tf.layers.batch_normalization(inputs=output, name='bn')
            output = tf.nn.lrn(input=output)
        if do_relu:
            output = tf.nn.leaky_relu(output, name='relu')
    return output


def max_pool(input, k_h, k_w, s_h, s_w, padding, name):
    with tf.variable_scope(name_or_scope=name):
        return tf.nn.max_pool(value=input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='mp')


def train():
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name='image')
    X = tf.reshape(image, shape=[batch_size, 28, 28, 1])
    Y = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 10], name='label')

    conv1, conv2, conv3, output = net(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))

    opt = tf.train.AdamOptimizer(0.01).minimize(loss)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    vars = tf.trainable_variables()
    for var in vars:
        print(var)

    with tf.Session() as sess:
        sess.run(init)
        if not os.path.exists(CKPT_VALID):
            for cnt in range(epoch):
                epoch_cost = 0
                train_imgs, train_labels = mnist.train.next_batch(batch_size)
                iter_cnt = mnist.train.num_examples // batch_size
                for i in range(iter_cnt):
                    _, cost = sess.run([opt, loss], feed_dict={image: train_imgs, Y: train_labels})
                    epoch_cost += cost
                logger.info('Epoch Loss: %.2f' % epoch_cost)
            saver.save(sess, CKPT)
            accuracy = sess.run(acc, feed_dict={image: mnist.test.images[:batch_size],
                                                Y: mnist.test.labels[:batch_size]})
            logger.info('ACC: %.2f' % accuracy)
        else:
            saver.restore(sess, CKPT)
            batch_num = mnist.test.num_examples // batch_size
            import random
            batch_idx = random.randint(0, batch_num - 1)
            print(batch_idx)
            image_conv1, image_conv2, image_conv3 = sess.run([conv1, conv2, conv3],
                                                             feed_dict=
                                                             {image:
                                                                  mnist.test.images[batch_idx * batch_size:
                                                                                    (batch_idx + 1) * batch_size]})

            random_idx = random.randint(0, batch_size - 1)
            print(random_idx)
            return np.array(mnist.test.images[batch_idx * batch_size + random_idx]).reshape(28, 28, 1) * 255, \
                   image_conv1[random_idx], \
                   image_conv2[random_idx], \
                   image_conv3[random_idx]


if __name__ == '__main__':
    train()
