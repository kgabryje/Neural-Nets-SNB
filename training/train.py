from __future__ import print_function
import numpy as np
import tensorflow as tf
from os import path
import pickle
import zipfile
import matplotlib
import sys
if sys.platform == 'linux':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image_size = 28
num_labels = 10
batch_size = 128
hidden_nodes = 500
num_steps = 5001
beta = 0.0005
init_learning_rate = 0.5
decay_steps = 800
decay_rate = 0.8
keep_probability = 0.9

data_path = path.join('..', 'data')
filelist = ['train_images', 'train_labels', 'validation_images', 'validation_labels', 'test_images', 'test_labels']
data_url = 'https://github.com/pplonski/datasets-for-start/tree/master/mnist'


# def build_digits_set(filename) -> (np.array, np.array):
#     if all([path.isfile(path.join(data_path, f + '.pickle')) for f in filelist]):
#         return
#     with open(filename) as f:
#         digits_set = np.array([[float(pixel) for pixel in digit.split(',')] for digit in f])
#     # return images, labels
#     return digits_set[:, 1:], (np.arange(num_labels) == digits_set[:, 0][:, None]).astype(np.float32)

def unzip_pickles():
    if not all([path.isfile(path.join(data_path, f + '.pickle')) for f in filelist]):
        print('unzipping...')
        zip_ref = zipfile.ZipFile(path.join(data_path, 'pickles.zip'), 'r')
        zip_ref.extractall(data_path)
        zip_ref.close()


def load_digits_set(filename):
    with open(path.join(data_path, filename), 'rb') as f:
        return pickle.load(f)

unzip_pickles()
test_images = load_digits_set('test_images.pickle')
test_labels = load_digits_set('test_labels.pickle')
train_images = load_digits_set('train_images.pickle')
train_labels = load_digits_set('train_labels.pickle')
validation_images = load_digits_set('validation_images.pickle')
validation_labels = load_digits_set('validation_labels.pickle')

# test_images, test_labels = build_digits_set('../data/test.csv')
# train_images, train_labels = build_digits_set('../data/train.csv')
# validation_images, validation_labels = build_digits_set('../data/validation.csv')


def generate_weights(shape):
    return tf.Variable(
        tf.truncated_normal(shape, stddev=0.1))


def generate_biases(shape):
    return tf.Variable(
        tf.random_uniform(shape, maxval=0.3))


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validation_images, dtype=tf.float32)
    tf_test_dataset = tf.constant(test_images, dtype=tf.float32)

    keep_prob = tf.placeholder(tf.float32)

    # Variables.
    weights_layer1 = generate_weights([image_size * image_size, hidden_nodes])
    biases_layer1 = generate_biases([hidden_nodes])

    weights_layer2 = generate_weights([hidden_nodes, hidden_nodes])
    biases_layer2 = generate_biases([hidden_nodes])

    weights_output = generate_weights([hidden_nodes, num_labels])
    biases_output = generate_biases([num_labels])

    global_step = tf.Variable(0, trainable=False)

    layer1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(tf_train_dataset, weights_layer1) + biases_layer1), keep_prob)
    layer2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(layer1, weights_layer2) + biases_layer2), keep_prob)
    logits = tf.nn.dropout(tf.matmul(layer2, weights_output) + biases_output, keep_prob)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)
        + beta*(tf.nn.l2_loss(weights_layer1) + tf.nn.l2_loss(weights_layer2) + tf.nn.l2_loss(weights_output)))

    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)

    valid_layer1 = tf.nn.tanh(tf.matmul(tf_valid_dataset, weights_layer1) + biases_layer1)
    valid_layer2 = tf.nn.tanh(tf.matmul(valid_layer1, weights_layer2) + biases_layer2)
    valid_logits = tf.matmul(valid_layer2, weights_output) + biases_output
    valid_prediction = tf.nn.softmax(valid_logits)

    test_layer1 = tf.nn.tanh(tf.matmul(tf_test_dataset, weights_layer1) + biases_layer1)
    test_layer2 = tf.nn.tanh(tf.matmul(test_layer1, weights_layer2) + biases_layer2)
    test_logits = tf.matmul(test_layer2, weights_output) + biases_output
    test_prediction = tf.nn.softmax(test_logits)

losses = []
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: keep_probability}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        losses.append(l)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), validation_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

plt.ion()
plt.plot(losses)
plt.show(block=True)
