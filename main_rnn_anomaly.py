import tensorflow as tf
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

##########################################
# Define "true" data generation process

def iterate(x, y, z):
    xnew = x + np.sin(y) + np.random.normal(0, np.exp(z), )
    ynew = y ** .5 + np.random.exponential(xnew**2, 1)[0]
    znew = .9 * z + .1 * np.random.normal(1., 0.1, 1)[0]
    return xnew, ynew, znew

T = 10000
xyz = np.zeros([T, 3])
xyz[0,:] = [1., 1., 1.]
for t in range(1,T):
    xyz[t,:] = iterate(xyz[t-1, 0], xyz[t-1, 1], xyz[t-1, 2])

print xyz[-5:,:]
tmp = plt.figure()
for j in range(3):
    tmp.add_subplot(3,1,j+1); plt.plot(xyz[:,j])
plt.savefig('temp/rnn_temp.png'); plt.close()

##########################################
# Define our RNN graph

batch_size = 25

