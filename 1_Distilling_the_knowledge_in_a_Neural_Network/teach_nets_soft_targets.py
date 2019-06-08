import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

from glob_params import MNIST_DATASET

mnist = input_data.read_data_sets(train_dir=MNIST_DATASET, one_hot=True)


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
            output = tf.divide(output, temperature)
        return output


def teacher_net(inputs, temperature):
    return net(inputs=inputs, temperature=temperature, net_name='teacher')


def student_net(inputs, temperature: int):
    return net(inputs=inputs, temperature=temperature, net_name='student')

def train():
    pass


if __name__ == '__main__':
    train()