import tensorflow as tf
import numpy as np
import copy

# Build graph
Xnp = np.random.normal(size=[100])
Bnp = 0.83
Ynp = Xnp * Bnp + np.random.normal(size=[100])

X = tf.constant(Xnp, dtype=tf.float32)
B = tf.Variable(dtype=tf.float32, initial_value=0.)
Y = tf.constant(Ynp, dtype=tf.float32)
Ypred = tf.multiply(X, B)
Loss = tf.losses.mean_squared_error(Y, Ypred)

# Add ops
train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(Loss)
init_op = tf.global_variables_initializer()  # keep this at the end of the list so it initializes the optimizer's variables

# Run
sess = tf.Session()
sess.run(init_op)
sess.run(Loss)
for i in range(100000):
    sess.run(train_op)
    if i % 1000 == 0:
        print sess.run(Loss)
print sess.run(Loss)