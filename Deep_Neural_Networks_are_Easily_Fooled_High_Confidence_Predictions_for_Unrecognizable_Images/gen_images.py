import sys

sys.path.append(r'F:\recoding_papers')

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from glob_params import MNIST_DATASET
import os
import numpy as np
import math

mnist = read_data_sets(train_dir=MNIST_DATASET, one_hot=True)

CKPT = os.path.join('.', 'gen_images.ckpt')
CKPT_VALID = os.path.join('.', 'checkpoint')

batch_size = 10
epoch = 2000


def net(inputs):
    with tf.variable_scope(name_or_scope='fc_gen', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.fully_connected],
                            num_outputs=50, normalizer_fn=slim.batch_norm, reuse=False):
            fc1 = slim.fully_connected(inputs=inputs, scope='fc1')
            fc2 = slim.fully_connected(inputs=fc1, scope='fc2')
            output = slim.fully_connected(inputs=fc2,
                                          num_outputs=10,
                                          scope='output',
                                          activation_fn=None,
                                          normalizer_fn=None)
        return output


def train(gen_image, target, make_pred):
    # tf.set_random_seed(1)
    tf.reset_default_graph()

    if os.path.exists(CKPT_VALID):
        do_train = False
    else:
        do_train = True

    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

    output = net(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
    opt = tf.train.AdamOptimizer().minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if do_train:
            iter_cnt = mnist.train.num_examples // batch_size
            for _ in range(epoch):
                epoch_cost = 0
                for i in range(iter_cnt):
                    train_images, train_labels = mnist.train.next_batch(batch_size)
                    _, cost = sess.run([opt, loss], feed_dict={X: train_images, Y: train_labels})
                    epoch_cost += cost
                print(epoch_cost)
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
            accuracy_student = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc = sess.run(accuracy_student, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print(acc)
            saver.save(sess, CKPT)
        else:
            saver.restore(sess, CKPT)
            if not make_pred:
                one_hot_tensor = sess.run(tf.one_hot(target, depth=10))
                one_hot_tensor = np.reshape(one_hot_tensor, newshape=[1, -1]).repeat(len(samples), axis=0)
                loss_not_sum = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y)
                cost = sess.run(loss_not_sum, feed_dict={X: gen_image, Y: one_hot_tensor})
                return cost
            else:
                pred = sess.run(output, feed_dict={X: gen_image})
                pred = np.argmax(pred, axis=1)
                return pred


if __name__ == '__main__':
    if not os.path.exists(CKPT_VALID):
        train(None, None, False)

    sample_num = 100
    valid_idx = [1] * sample_num
    samples = np.random.randint(low=0, high=256, size=[sample_num, 784]) / 255
    clean_rate = 0.01
    cross_rate = 0.4
    keep_rate = 0.05
    while sum(valid_idx) >= int(keep_rate * sample_num):
        print(sum(valid_idx))
        costs = train(samples, 1, False)
        costs = np.reshape(costs, newshape=[sample_num, -1])
        costs = map(sum, costs)
        costs = list(costs)
        print(min(costs))

        clean_sample = math.ceil(clean_rate * sum(valid_idx))
        sorted_costs = sorted(costs)
        to_be_cleaned = sorted_costs
        clean_cnt = 0
        for tbc in reversed(to_be_cleaned):
            if clean_cnt == clean_sample:
                break
            cleaned_idx = costs.index(tbc)
            if valid_idx[cleaned_idx] == 1:
                valid_idx[cleaned_idx] = 0
                clean_cnt += 1

        cross_sample = int(cross_rate // 2 * sum(valid_idx))
        to_be_crossed = sorted_costs[:cross_sample * 2]
        for idx in range(cross_sample):
            cross_idx1 = costs.index(to_be_crossed[2 * idx])
            cross_idx2 = costs.index(to_be_crossed[2 * idx + 1])
            if valid_idx[cross_idx1] == 1 and valid_idx[cross_idx2] == 1:
                cross_sample1 = samples[cross_idx1]
                cross_sample2 = samples[cross_idx2]

                cross_start = 1
                cross_end = 0
                while cross_start > cross_end - 100:
                    cross_start = np.random.randint(low=0, high=785)
                    cross_end = np.random.randint(low=0, high=785)

                for cidx in range(cross_start, cross_end):
                    temp = cross_sample1[cidx]
                    cross_sample1[cidx] = cross_sample2[cidx]
                    cross_sample2[cidx] = temp

        mutation = np.random.randint(low=0, high=11, size=[sample_num, 784])
        mutation_sample = np.random.randint(low=0, high=11, size=[sample_num])
        for idx in range(len(mutation_sample)):
            if valid_idx[idx] == 1:
                if mutation_sample[idx] > 0:
                    m_rate = mutation[idx]
                    for mr_idx in range(len(m_rate)):
                        if m_rate[mr_idx] > 0:
                            samples[idx][mr_idx] = 1 - samples[idx][mr_idx]
    max_cost = max(costs)
    pred = train(np.reshape(samples[costs.index(max_cost)], newshape=[1, -1]).repeat(1000, axis=0), None, True)
    print(pred)
    import cv2

    cv2.imwrite('5.jpg', np.reshape(samples[costs.index(max_cost)] * 255, newshape=[28, 28, 1]))
