import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

save_file = './model.ckpt'

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

test_data=[[73., 66., 70.]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# save model
# saver = tf.train.Saver()

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# weight 및 bias 저장
saver = tf.train.Saver()

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)
    # if you want training
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

    # 결과값 확인
    result = sess.run(hypothesis, feed_dict={X: test_data})
    print("result:", result)
    saver.save(sess, save_file)
