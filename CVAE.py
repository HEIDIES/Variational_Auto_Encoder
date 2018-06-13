# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/6 10:13
# @Author  : Yuzhou Hou
# @Email   : m18010639062@163.com
# @File    : CVAE.py
# Description : Conditional variational auto encoder
# --------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data", one_hot = True)

n_input = 784
n_labels = 10
n_hidden_1 = 256
n_hidden_2 = 2

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_labels])
zinput = tf.placeholder(tf.float32, [None, n_hidden_2])

weights = {
    'w1' : tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev = 0.001)),
    'label_w1' : tf.Variable(tf.truncated_normal([n_labels, n_hidden_1], stddev = 0.001)),
    'mean_w1' : tf.Variable(tf.truncated_normal([2 * n_hidden_1, n_hidden_2], stddev = 0.001)),
    'log_sigma_w1' : tf.Variable(tf.truncated_normal([2 * n_hidden_1, n_hidden_2], stddev = 0.001)),
    'w2' : tf.Variable(tf.truncated_normal([n_hidden_2 + n_labels, n_hidden_1], stddev = 0.001)),
    'w3' : tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev = 0.001)),
    
    'b1' : tf.Variable(tf.zeros([n_hidden_1])),
    'label_b1' : tf.Variable(tf.zeros([n_hidden_1])),
    'mean_b1' : tf.Variable(tf.zeros([n_hidden_2])),
    'log_sigma_b1' : tf.Variable(tf.zeros([n_hidden_2])),
    'b2' : tf.Variable(tf.zeros([n_hidden_1])),
    'b3' : tf.Variable(tf.zeros([n_input]))
}

l1_encode_x = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
l1_encode_y = tf.nn.relu(tf.add(tf.matmul(y, weights['label_w1']), weights['label_b1']))

l1_out = tf.concat([l1_encode_x, l1_encode_y], 1)
l2_mean = tf.matmul(l1_out, weights['mean_w1']) + weights['mean_b1']
l2_log_sigma = tf.matmul(l1_out, weights['log_sigma_w1']) + weights['log_sigma_b1']

eps = tf.random_normal(tf.stack([tf.shape(l1_encode_x)[0], n_hidden_2]), 0, 1, dtype = tf.float32)
z = tf.add(l2_mean, tf.multiply(tf.sqrt(tf.exp(l2_log_sigma)), eps))

z_all = tf.concat([z, y], 1)
l1_decode = tf.nn.relu(tf.add(tf.matmul(z_all, weights['w2']), weights['b2']))
out = tf.matmul(l1_decode, weights['w3']) + weights['b3']

zinput_all = tf.concat([zinput, y], 1)
l1_decode_out = tf.nn.relu(tf.add(tf.matmul(zinput_all, weights['w2']), weights['b2']))
out_out = tf.matmul(l1_decode_out, weights['w3']) + weights['b3']

reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(out - x, 2))
latent_loss = -0.5 * tf.reduce_sum(1 + l2_log_sigma - tf.square(l2_mean) - tf.exp(l2_log_sigma), 1)
loss_function = tf.reduce_mean(reconstruction_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.002).minimize(loss_function)

epochs = 100
batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss = sess.run([optimizer, loss_function], feed_dict = {x : batch_x, y : batch_y})
        cost += loss
    if (epoch + 1) % 5 == 0:
        print("epoch %02d/%02d, cost = %.9f" %(epoch + 1, epochs, cost / total_batch))
print("finished!")
print("cost = %.9f" %(sess.run(loss_function, feed_dict = {x : mnist.test.images, y : mnist.test.labels})))

z_sample = np.random.randn(10, 2) * 0
pred = sess.run(out_out , feed_dict = {zinput : z_sample, y : mnist.test.labels[:10]})

f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(pred[i], (28, 28)))
plt.show()

#sess.close()
