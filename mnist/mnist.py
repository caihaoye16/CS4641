import pandas as pd
import numpy as np
import csv as csv
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


train_df = pd.read_csv('train.csv', header=0) 
test_df = pd.read_csv('test.csv', header=0) 

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# y_ = tf.Print(y_, [x, y_, tf.shape(x), tf.shape(y_)], message = "y_:")


def weight_variable(shape):
  	initial = tf.truncated_normal(shape, stddev=0.1)
  	return tf.Variable(initial)

def bias_variable(shape):
  	initial = tf.constant(0.1, shape=shape)
  	return tf.Variable(initial)

def conv2d(x, W):
  	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# W_conv1 = tf.Print(W_conv1, [W_conv1, x, y_, tf.shape(x), tf.shape(y_)], message = "Wconv1:", summarize = 50)

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# y_conv = tf.Print(y_conv, [W_conv1, x, y_, tf.shape(x), tf.shape(y_)], message = "y_conv:")


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# cross_entropy = tf.Print(cross_entropy, [W_conv1, x, y_, tf.shape(x), tf.shape(y_)], message = "cross:")

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# W_conv1 = tf.Print(W_conv1, [W_conv1, x, y_, tf.shape(x), tf.shape(y_)], message = "Wconv1:", summarize = 50)


prediction = tf.argmax(y_conv,1)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
# print sess.run(W_conv1)

label = train_df["label"].values

y_label = []
for i in label:
	temp = [0,0,0,0,0,0,0,0,0,0]
	temp[i] = 1
	y_label.append(temp)

temp_y = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
print len(label)
# for i in range(0, len(label), 50):
# #   	# batch = mnist.train.next_batch(50)
# 	batch = train_df.drop(["label"], axis = 1).values[i:i+50]
# 	batch = batch / 255.0
# 	print i
# 	# print batch[0]
# 	# print np.array(y_label[i:i+50]).shape
#   	train_step.run(feed_dict={x: batch, y_: np.array(y_label[i:i+50]).astype(float), keep_prob: 0.8})
#   	# print sess.run(W_conv1)

for i in range(20000):
  	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	# print batch[0][1]
  	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_x = test_df.values
output = np.array(prediction.eval(feed_dict={x: np.array([test_x[0]]).astype(float), y_: temp_y, keep_prob: 1.0}))
for i in range(1,len(test_x)):
	temp = prediction.eval(feed_dict={x: np.array([test_x[i]]).astype(float), y_: temp_y, keep_prob: 1.0})
	# print output
	output = np.concatenate((output, temp), axis = 0)

print(output)

predictions_file = open("myfirstcnn.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","label"])
for i in range(len(output)):
	open_file_object.writerow([i,output[i]])
predictions_file.close()