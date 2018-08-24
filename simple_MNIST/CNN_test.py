# -*- coding: utf-8 -*-

import tensorflow as tf
from simple_MNIST import input_data

"""
    简单的MNIST手写数字识别模型CNN。
    来源：http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
"""

class simple_CNN(object):
    def __init__(self):
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", [None, 10])
        self.keep_prob = tf.placeholder("float")
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_varible(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding='SAME')

    def conv1(self):
        W_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_varible([32])
        x_image = tf.reshape(self.x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        return h_pool1

    def conv2(self):
        W_conv2 = self.weight_variable([5,5,32,64])
        b_conv2 = self.bias_varible([64])
        h_pool1 = self.conv1()

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        return h_pool2

    def dense_con(self):
        W_fc1 = self.weight_variable([7*7*64, 1024])
        b_fc1 = self.bias_varible([1024])
        h_pool2 = self.conv2()

        h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
        return h_fc1

    def dropout(self):
        h_fc1 = self.dense_con()
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        return h_fc1_drop

    def output(self):
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_varible([10])
        h_fc1_drop = self.dropout()

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
        return y_conv

    def loss(self):
        y_conv = self.output()
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(y_conv))
        return cross_entropy

    def accuracy(self):
        cross_entropy = self.loss()
        correct_prediction = tf.equal(tf.argmax(cross_entropy, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy

    def train(self):
        train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(self.loss())
        accuracy = self.accuracy()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = self.mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g" %(i, train_accuracy))
            train_step.run(session=sess, feed_dict={self.x: batch[0], self.y_: batch[1],
                                      self.keep_prob: 0.5})
        accuracy.eval(session=sess, feed_dict={self.x: self.mnist.test.images,
                                        self.y_: self.mnist.test.labels,
                                        self.keep_prob: 1.0})



if __name__ == "__main__":
    cnn = simple_CNN()
    cnn.train()



