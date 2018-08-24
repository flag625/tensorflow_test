# -*- coding: utf-8 -*-

import tensorflow as tf
from simple_MNIST import input_data

"""
    简单的MNIST手写数字识别模型。
    来源：http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
"""

class simple_MNIST(object):
    def __init__(self):
        self.x = tf.placeholder("float", [None, 784])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W)+self.b)
        self.y_ = tf.placeholder("float", [None, 10])
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch_xs, batch_ys = self.mnist.train.next_batch(100)
                sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuarcy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print(sess.run(accuarcy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}))

if __name__ == "__main__":
    mnist = simple_MNIST()
    mnist.train()




