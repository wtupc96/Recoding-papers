import sys
import time

sys.path.append(r'F:\recoding_papers')

import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import logging
from glob_params import MNIST_DATASET

mnist = read_data_sets(train_dir=MNIST_DATASET, one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', '-T', type=float, default=-1,
                    help='The temperature for DISTILLING.')
parser.add_argument('--batch_size', '-B', type=int, default=1000,
                    help='The batch size of taking images in.')
parser.add_argument('--epoch_teacher', '-ET', type=int, default=500,
                    help='The training epoch for teacher net.')
parser.add_argument('--epoch_student_individual', '-ESI', type=int, default=500,
                    help='The individual training epoch for student net.')
parser.add_argument('--epoch_student_learning', '-ESL', type=int, default=100,
                    help='The learning from teacher training epoch for student net.')

args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def net(inputs, net_name: str):
    """
    Network architecture template.
    :param inputs: Network input
    :param net_name: Network name for var_scope
    :return: Network output
    """
    if 'teacher' in net_name:
        num_output = 50
        norm_fn = slim.batch_norm
        do_dropout = False
    elif 'student' in net_name:
        num_output = 30
        norm_fn = slim.batch_norm
        do_dropout = False
    else:
        raise Exception('Unrecognized NET.')

    with tf.variable_scope(name_or_scope=net_name, reuse=False):
        with slim.arg_scope([slim.fully_connected],
                            num_outputs=num_output, normalizer_fn=norm_fn):
            fc1 = slim.fully_connected(inputs=inputs, scope='fc1')
            fc2 = slim.fully_connected(inputs=fc1, scope='fc2')
            output = slim.fully_connected(inputs=fc2,
                                          num_outputs=10,
                                          scope='output',
                                          activation_fn=None,
                                          normalizer_fn=None)
            if do_dropout:
                output = slim.dropout(inputs=output, keep_prob=0.9)
        return output


def teacher_net(inputs, identifier: str):
    """
    Build the Teacher Network.
    :param inputs: Teacher Network inputs
    :param identifier: The one and only identifier for Teacher Network
    :return: Teacher Network outputs
    """
    return net(inputs=inputs, net_name='teacher' + identifier)


def student_net(inputs, identifier: str):
    """
    Build the Student Network.
    :param inputs: Student Network inputs
    :param identifier: The one and only identifier for Student Network
    :return: Student Network outputs
    """
    return net(inputs=inputs, net_name='student' + identifier)


def train(temperature: float,
          batch_size: int,
          epoch_teacher: int,
          epoch_student_individual: int,
          epoch_student_learning: int):
    """
    Train Teacher and Student Network.
    :param temperature: Temperature for distilling
    :param batch_size: batch size for MNIST
    :param epoch_teacher: Training epoch of Teacher Network
    :param epoch_student_individual: Training epoch of Student Network
    :param epoch_student_learning: Teaching epoch of Student Network
    :return: Enhancement of the accuracy of Student Network
    """
    tf.set_random_seed(1)

    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

    identifier = str(time.time())
    output_teacher = teacher_net(X, identifier)
    output_student = student_net(X, identifier)

    loss_teacher = slim.losses.softmax_cross_entropy(logits=output_teacher, onehot_labels=Y, scope='loss_teacher')
    loss_student = slim.losses.softmax_cross_entropy(logits=output_student, onehot_labels=Y, scope='loss_student')

    loss_student_learning = temperature ** 2 * slim.losses.mean_squared_error(
        predictions=slim.softmax(output_student / temperature),
        labels=slim.softmax(output_teacher / temperature))

    vars = tf.trainable_variables()

    vars_teacher = [v for v in vars if 'teacher' + identifier in v.name]
    vars_student = [v for v in vars if 'student' + identifier in v.name]

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    opt_teacher = optimizer.minimize(loss_teacher, var_list=vars_teacher, name='loss_teacher')
    opt_student = sgd_optimizer.minimize(loss_student, var_list=vars_student, name='loss_student')
    opt_student_learning = sgd_optimizer.minimize(loss_student_learning, name='loss_student')

    correct_prediction_teacher = tf.equal(tf.argmax(Y, 1), tf.argmax(output_teacher, 1))
    correct_prediction_student = tf.equal(tf.argmax(Y, 1), tf.argmax(output_student, 1))

    accuracy_teacher = tf.reduce_mean(tf.cast(correct_prediction_teacher, tf.float32))
    accuracy_student = tf.reduce_mean(tf.cast(correct_prediction_student, tf.float32))

    init_vars = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_vars)

        logger.info('Student Net is learning.')
        for cnt in range(epoch_student_individual):
            train_images, train_labels = mnist.train.next_batch(batch_size)
            _, cost_student = sess.run([opt_student, loss_student],
                                       feed_dict={X: train_images, Y: train_labels})
            if cnt % (epoch_student_individual // 5 + 1) == 0:
                logger.info('Student loss: {}'.format(cost_student))
        acc_student_individual = sess.run(accuracy_student,
                                          feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        logger.info('Student INDIVIDUAL Accuarcy: {}'.format(acc_student_individual))

        logger.info('Teacher Net is learning.')
        for cnt in range(epoch_teacher):
            train_images, train_labels = mnist.train.next_batch(batch_size)
            _, cost_teacher = sess.run([opt_teacher, loss_teacher],
                                       feed_dict={X: train_images, Y: train_labels})

            _, cost_student = sess.run([opt_student, loss_student],
                                       feed_dict={X: train_images, Y: train_labels})
            if cnt % (epoch_teacher // 5 + 1) == 0:
                logger.info('Teacher loss: {}'.format(cost_teacher))
        acc_teacher = sess.run(accuracy_teacher,
                               feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        logger.info('Teacher Accuracy: {}'.format(acc_teacher))

        logger.info('Teacher is teaching Student.')
        for cnt in range(epoch_student_learning):
            train_images, _ = mnist.train.next_batch(batch_size)
            _, cost_student = sess.run([opt_student_learning, loss_student_learning],
                                       feed_dict={X: train_images})
            if cnt % (epoch_student_learning // 5 + 1) == 0:
                logger.info('Student loss: {}'.format(cost_student))
        acc_student_learning = sess.run(accuracy_student,
                                        feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        logger.info('Student LEARNING Accuarcy: {}'.format(acc_student_learning))
        enhancement = acc_student_learning - acc_student_individual
        logger.info('Accuaracy gained by: {}'.format(enhancement))
        return enhancement


if __name__ == '__main__':
    logger.info(args)
    temperature = args.temperature
    batch_size = args.batch_size
    epoch_teacher = args.epoch_teacher
    epoch_student_individual = args.epoch_student_individual
    epoch_student_learning = args.epoch_student_learning

    # If temperature > 0, calculate the enhancement under cmd params; else draw the fig.
    if temperature > 0:
        logger.info('Calc the enhancement.')
        train(temperature, batch_size, epoch_teacher, epoch_student_individual, epoch_student_learning)
    else:
        import numpy as np
        from matplotlib import pyplot as plt

        logger.info('Draw the relationship between T and according enhancement.')
        x = np.linspace(-10, 100, num=300)
        y = list()
        for temperature in x:
            y.append(train(temperature, batch_size, epoch_teacher, epoch_student_individual, epoch_student_learning))

        plt.xlabel('T')
        plt.ylabel('Enhancement %')
        idx = y.index(max(y))
        plt.title('MAX enhancement={}, at T={}'.format(y[idx], x[idx]))
        plt.plot(x, np.array(y) * 100)
        plt.savefig('./result.png')
        plt.show()
