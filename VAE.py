# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/5 9:00
# @Author  : Yuzhou Hou
# @Email   : m18010639062@163.com
# @File    : VAE.py
# Description : Variational auto encoder
# --------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/") # one_hot = True

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 2  # 2-dimensional Gaussian mixture distribution, change it to n for n-dim GMM

x = tf.placeholder(tf.float32, [None, n_input])
zinput = tf.placeholder(tf.float32, [None, n_hidden_2]) # Allow us to generate image by
# inputting a normal distribution N(mean, sigma)

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.001)),
    'mean_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'log_sigma_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.001)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.001)),

    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'mean_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'log_sigma_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),
    'b3': tf.Variable(tf.zeros([n_input]))
}

l2_encoder = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
l2_mean = tf.add(tf.matmul(l2_encoder, weights['mean_w1']), weights['mean_b1'])
l2_log_sigma = tf.add(tf.matmul(l2_encoder, weights['log_sigma_w1']), weights['log_sigma_b1'])

eps = tf.random_normal(tf.stack([tf.shape(l2_encoder)[0], n_hidden_2]), 0, 1, dtype = tf.float32)

# eps ~ N(0,1), z = mean + eps * sigma ~ N(mean,sigma)
z = tf.add(l2_mean, tf.multiply(tf.sqrt(tf.exp(l2_log_sigma)), eps))
l1_encoder = tf.nn.relu(tf.add(tf.matmul(z, weights['w2']), weights['b2']))
out = tf.matmul(l1_encoder, weights['w3']) + weights['b3']

l1_encoder_out = tf.nn.relu(tf.add(tf.matmul(zinput, weights['w2']), weights['b2']))
out_out = tf.matmul(l1_encoder_out, weights['w3']) + weights['b3']

reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(out, x), 2))

#KL divergence
latent_loss = -0.5 * tf.reduce_sum(1 + l2_log_sigma - tf.square(l2_mean) - tf.exp(l2_log_sigma), 1)
loss_function = tf.reduce_mean(reconstruction_loss + latent_loss)


#optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_function)
optimizer = tf.train.AdamOptimizer(0.002).minimize(loss_function)

#epochs = 50
epochs = 100
batch_size = 128
display_step = 5

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss = sess.run([optimizer, loss_function], feed_dict = {x : batch_x})
        cost += loss
    if (epoch + 1) % display_step == 0:
        print("epoch %02d/%02d cost = %.6f" %(epoch + 1, epochs, cost / total_batch))

print("finished!")
loss = sess.run(loss_function, feed_dict={x : mnist.test.images})
print("result : ", loss)

test = mnist.test.images[:10]
rec = sess.run(out, feed_dict = {x : test})
f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(rec[i], (28, 28)))
plt.show()

aa = [l for l in mnist.test.labels]
z_out = sess.run(z, feed_dict = {x : mnist.test.images})
plt.scatter(z_out[:,0], z_out[:,1], c = aa)
plt.colorbar()
plt.show()

n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_out = sess.run(out_out, feed_dict={zinput: z_sample})

        digit = x_out[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

#sess.close()
