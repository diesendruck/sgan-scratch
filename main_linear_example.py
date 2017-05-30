import tensorflow as tf
import numpy as np
import copy

# Build graph
n = 100
p = 1
Xnp = np.random.normal(size=[n, p])
Bnp = 0.83
Ynp = Xnp * Bnp + np.random.normal(size=[n, 1])

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
for i in range(10000):
    sess.run(train_op)
    if i % 1000 == 0:
        print sess.run(Loss)
print sess.run(Loss)



########################################
# Again, with feed_dict's!
batch_size = 25
selector = tf.placeholder(X.dtype, [batch_size, n], name='selector')
Xb = tf.matmul(selector, X)
# B is already defined above
Yb = tf.matmul(selector, Y)
Ypredb = tf.multiply(Xb, B)
Lossb = tf.losses.mean_squared_error(Yb, Ypredb)
train_op_b = tf.train.AdamOptimizer(learning_rate=0.01).minimize(Lossb)
init_op_b = tf.global_variables_initializer()

# Run
np.random.choice(range(n), batch_size)
sess = tf.Session()
sess.run(init_op)
sess.run(Lossb, feed_dict={selector:np.identity(n)[np.random.choice(range(n), batch_size),:]})
for i in range(10000):
    sess.run(train_op, feed_dict={selector:np.identity(n)[np.random.choice(range(n), batch_size),:]})
    if i % 1000 == 0:
        print sess.run([Loss, Lossb], feed_dict={selector:np.identity(n)[np.random.choice(range(n), batch_size),:]})
print sess.run([Loss, Lossb], feed_dict={selector: np.identity(n)[np.random.choice(range(n), batch_size), :]})
