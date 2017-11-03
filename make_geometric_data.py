img_dir = '../data/geome/'
feature_file = 'features.txt'
n_trn = 6000
n_vld = 2000
n_tst = 2000
n_images = 10000

###########################################################
import numpy as np
###########################################################


color_rand = np.random.randint(0, 3, n_images)
for n in range(n_images):