import tensorflow as tf
from tensorflow.contrib import rnn
import pickle

#import dataset
with open('database.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)
print(train_x[0])

#hyperparameters
#common
learning_rate = 0.001
batch_size = 128
training_steps = 1000
display_step = 20
n_classes = 2

#for cnn
image_size = 76800 #320x240

#for rnn
n_input = 153600 # 80x60x32
n_hidden = 512
time_steps = 6

X = tf.placeholder(tf.float32, [None, time_steps, image_size])
y = tf.placeholder(tf.float32, [None, n_classes])

W = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
}

b = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.add(x, b)
    return tf.nn.relu(x)


def max_pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def feed_forward_cnn(x, W, b):

    x = tf.reshape(x, [-1, 320, 240, 1])

    conv1 = conv2d(x, W['wc1'], b['bc1'])
    conv1 = max_pool(conv1)

    conv2 = conv2d(conv1, W['wc2'], b['bc2'])
    conv2 = max_pool(conv2)

    return conv2


def feed_forward_rnn(input, W, b):

    input = tf.reshape(input, [-1, time_steps, n_input])
    input = tf.unstack(input, time_steps, 1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, state = rnn.static_rnn(lstm_cell, input, dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1], W['out']), b['out'])


def cnn_rnn(inputs, W, b):
    outs_cnn = []
    # inputs - (batch size, time_steps, image_size)
    inputs = tf.unstack(inputs, time_steps, 1)

    for i in range(time_steps):
        outs_cnn.append(feed_forward_cnn(inputs[i], W, b))

    return feed_forward_rnn(outs_cnn, W, b)


logits = cnn_rnn(X, W, b)
prediction = tf.nn.softmax(logits)
print(prediction)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

correct_pred = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    current_batch = 0

    for step in range(1, training_steps + 1):
        batch_x = train_x[current_batch: current_batch + batch_size]
        batch_y = train_y[current_batch: current_batch + batch_size]
        current_batch += batch_size
        if current_batch >= len(train_x):
            current_batch = 0

        sess.run(train, {X: batch_x, y: batch_y})

        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], {X: batch_x, y: batch_y})
            print("prediction", sess.run(prediction, {X: batch_x, y: batch_y}))
            print("step ", step, "of ", training_steps, ", loss = ", loss, ", accuracy = ", acc)


    print("Test accuracy is ", sess.run(accuracy, {X: test_x, y: test_y}))



