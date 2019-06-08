import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

from glob_params import MNIST_DATASET

mnist = input_data.read_data_sets(train_dir=MNIST_DATASET, one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', '-T', type=int, required=True, help='The temperature for DISTILLING.')
parser.add_argument('--batch_size', '-B', type=int, default=1000, help='The batch size of taking images in.')
parser.add_argument('--epoch', '-E', type=int, default=10000, help='The training epoch.')
args = parser.parse_args()


def net(inputs, temperature: int, net_name: str):
    if net_name == 'teacher':
        num_output = 1200
        norm_fn = slim.batch_norm
        do_dropout = True
    elif net_name == 'student':
        num_output = 800
        norm_fn = None
        do_dropout = False
    else:
        raise Exception('Unrecognized NET.')

    with tf.name_scope(name=net_name):
        with slim.arg_scope([slim.fully_connected], num_output=num_output, normalizer_fn=norm_fn):
            fc1 = slim.fully_connected(inputs=inputs, scope='fc1')
            fc2 = slim.fully_connected(inputs=fc1, scope='fc2')
            output = slim.fully_connected(inputs=fc2, num_outputs=10, scope='output')
            if do_dropout:
                output = slim.dropout(inputs=output)
            # Distilling
            output = tf.divide(output, temperature)
        return output


def teacher_net(inputs, temperature):
    return net(inputs=inputs, temperature=temperature, net_name='teacher')


def student_net(inputs, temperature: int):
    return net(inputs=inputs, temperature=temperature, net_name='student')


def train():
    temperature = args.temperature
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')
    output4acc = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='output4acc')

    output_teacher = teacher_net(X, temperature)
    output_student = student_net(X, temperature)

    loss_teacher = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output_teacher, logits=Y)
    loss_student = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output_student, logits=Y)

    optimizer = tf.train.AdamOptimizer()
    opt_teacher = optimizer.minimize(loss_teacher)
    opt_student = optimizer.minimize(loss_student)

    correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(output4acc, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        for cnt in range(args.epoch):
            train_images, train_labels = mnist.train.next_batch(args.batch_size)
            cost_student, _ = sess.run([loss_student, opt_student], feed_dict={X: train_images, Y: train_labels})
            cost_teacher, _ = sess.run([loss_teacher, opt_teacher], feed_dict={X: train_images, Y: train_labels})
            if cnt % 100 == 0:
                print(cost_student)
                print(cost_teacher)
                print()
        acc_teacher = sess.run(accuracy,
                               feed_dict={X: mnist.test.images, Y: mnist.test.labels, output4acc: output_teacher})
        acc_student = sess.run(accuracy,
                               feed_dict={X: mnist.test.images, Y: mnist.test.labels, output4acc: output_student})
        print(acc_teacher)
        print(acc_student)


if __name__ == '__main__':
    train()
