import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/", one_hot = True)

slim = tf.contrib.slim

n_inputs = 784
n_labels = 10
n_latents = 10

# Leaky relu
def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.01 * x)

x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_labels])
zinput = tf.placeholder(tf.float32, [None, n_latents])

net = tf.reshape(x, [-1, 28, 28, 1])
net = slim.conv2d(net, 64, kernel_size = [4, 4], stride = 2, activation_fn = leaky_relu)
net = slim.conv2d(net, 128, kernel_size = [4, 4], stride = 2, activation_fn = leaky_relu)
net = slim.flatten(net)
net = slim.fully_connected(net, 128, activation_fn = leaky_relu)

net_1 = slim.fully_connected(y, 128)
net_all = tf.concat([net, net_1], axis = 1)

mean = slim.fully_connected(net_all, n_latents)
log_sigma = slim.fully_connected(net_all, n_latents)

eps = tf.random_normal(tf.stack([tf.shape(net_all)[0], n_latents]), 0, 1, dtype = tf.float32)
z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma)), eps))

# Reuse these weights for generator
def decoder(z, y):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('decoder')]) > 0
    with tf.variable_scope('decoder', reuse = reuse):
        z_all = tf.concat([z, y], axis = 1)
        out = slim.fully_connected(z_all, 128)
        out = slim.batch_norm(out, activation_fn = tf.nn.relu)
        out = slim.fully_connected(out, 7 * 7 * 128)
        out = slim.batch_norm(out, activation_fn = tf.nn.relu)
        out = tf.reshape(out, [-1, 7, 7, 128])
        out = slim.conv2d_transpose(out, 64, kernel_size = [4, 4], stride = 2)
        out = slim.batch_norm(out, activation_fn = tf.nn.relu)
        out = slim.conv2d_transpose(out, 1, kernel_size = [4, 4], stride = 2, activation_fn = tf.nn.sigmoid)
    return out

out = decoder(z, y) # Reconstructor
z_out = decoder(zinput,y) # Generator
out = tf.reshape(out, [-1, 784])
reconstruction_loss = 0.5 * tf.reduce_sum(tf.square(out - x))

# KL divergence
latent_loss = - 0.5 * tf.reduce_sum(1 - tf.square(mean) + log_sigma - tf.exp(log_sigma), 1)
loss_function =  tf.reduce_mean(reconstruction_loss + latent_loss)

# Change the learning rate
optimizer = tf.train.AdamOptimizer(0.002).minimize(loss_function)

epochs = 3
batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Can add some noise
        feeds = {x : batch_x, y : batch_y}
        _, loss = sess.run([optimizer, loss_function], feeds)
        cost += loss
    print("epoch %02d/%02d, cost = %.9f" %(epoch + 1, epochs, cost / total_batch))
print("finished!")
print("cost = %.9f" %(sess.run(loss_function, feed_dict = {x : mnist.test.images, y : mnist.test.labels})))

# Test the reconstuctor(decoder)
pred = sess.run(out , feed_dict = {x : mnist.test.images[:10], y : mnist.test.labels[:10]})
f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(pred[i], (28, 28)))
plt.show()

# Test the generator
z_sample = np.random.randn(10, 10) * 0.1 # Change this scale factor and add some mean
pred_p = sess.run(z_out , feed_dict = {zinput : z_sample, y : mnist.test.labels[:10]})

f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(pred_p[i], (28, 28)))
plt.show()