import sys

sys.path.append('/home/se-oypc/recoding_papers')

import os
import tensorflow as tf
from glob_params import MNIST_DATASET, logger
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

batch_size = 1000
epoch = 100

mnist = read_data_sets(train_dir=MNIST_DATASET, one_hot=True)

CKPT = os.path.join('.', 'maxout.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')


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


def maxout(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
                         .format(num_channels, num_units))
    shape[axis] = -1
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def conv2d(input, output_ch, f_h, f_w, s_h, s_w, padding, do_norm, do_maxout, name):
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
            output = tf.layers.batch_normalization(inputs=output, name='bn')
        if do_maxout:
            output = maxout(inputs=output, num_units=output_ch // 2, axis=None)
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
                logger.info('Epoch Loss: %.4f' % epoch_cost)
            saver.save(sess, CKPT)

            cnt = len(mnist.test.labels) // batch_size
            accuracy = 0
            for _ in range(cnt):
                accuracy_ = sess.run(acc, feed_dict={image: mnist.test.images[:batch_size],
                                                     Y: mnist.test.labels[:batch_size]})
                accuracy += accuracy_
            accuracy = accuracy / cnt
            logger.info('ACC: %.4f' % accuracy)
        else:
            saver.restore(sess, CKPT)

            cnt = len(mnist.test.labels) // batch_size
            accuracy = 0
            for _ in range(cnt):
                accuracy_ = sess.run(acc, feed_dict={image: mnist.test.images[:batch_size],
                                                     Y: mnist.test.labels[:batch_size]})
                accuracy += accuracy_
            accuracy = accuracy / cnt
            logger.info('ACC: %.4f' % accuracy)


if __name__ == '__main__':
    train()
