img_dir = '../data/geome/'
feature_file = 'features.txt'
image_size = 128
patch_size = 96
n_trn = 600
n_vld = 200
n_tst = 200
n_images = n_trn + n_vld + n_tst
n_colors = 4
n_bg_colors = n_colors + 0
n_shapes = 4
channels = 3
shape_type = ['rectangles', 'polygons', 'stars', 'simple'][3]
inflation_rate = 100
random_rotation = False
noise_scale = 0.03

###########################################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import imsave
import os

def base_shape_points(shape_num, random_rotation = True, mode = ['rectangles', 'polygons', 'stars'][0]):
    if random_rotation:
        offset = np.random.random() * 2 * np.pi
    else:
        offset = 0.
    if mode == 'rectangles' or 'simple':
        if shape_num % 2 == 0:
            return [(np.cos(theta), np.sin(theta)) for theta in sorted([offset + deg * np.pi / 180. for deg in [20, 160, 200, 340]])]
        elif shape_num % 2 == 1:
            return [(np.cos(theta), np.sin(theta)) for theta in sorted([offset + deg * np.pi / 180. for deg in [70, 110, 250, 290]])]
    if mode == 'polygons':
        return [(np.cos(theta), np.sin(theta)) for theta in sorted([offset + deg * np.pi / 180. for deg in [idx * 360. / (shape_num + 3.) for idx in xrange(shape_num + 3)]])]
    if mode == 'stars':
        thetas = [offset + deg * np.pi / 180. for deg in [idx * 360. / (shape_num + 2.) / 2. for idx in xrange((shape_num + 2) * 2)]]
        rs = [.33 + .67 * (idx % 2) for idx in xrange((shape_num + 2) * 2)]
        return [(r * np.cos(theta), r * np.sin(theta)) for theta, r in sorted(zip(thetas, rs))]

def interpolate(x, y, factor):
    return (x[0] + (y[0] - x[0]) * factor, x[1] + (y[1] - x[1]) * factor)

def convert_points_to_patch(points, patch_size = 64, inflation_rate = 100):
    # assumes that points is a list of angle-ordered 2d coordinate tuples 
    mat = np.zeros([patch_size, patch_size])
    inflated_points = [interpolate(points[idx], points[(idx+1) % len(points)], inflator / (inflation_rate + 0.)) 
                       for inflator in xrange(inflation_rate) for idx in range(len(points))]
    thetas, rs = zip(*sorted([(np.arctan2(x, y), (x ** 2 + y ** 2) ** .5 * patch_size / 2.) for x,y in inflated_points]))
    for i in xrange(patch_size):
        for j in xrange(patch_size):
            y = i - patch_size / 2. + .5
            x = j - patch_size / 2. + .5
            theta, r = (np.arctan2(x, y), (x ** 2 + y ** 2) ** .5)
            idx = np.array(thetas).searchsorted(theta) % len(rs)
            val = max(0., min(1., rs[idx] - r + .5))
            mat[i, j] = val
    return mat


# ######################################################################################################################
# for shape_type in ['rectangles', 'polygons', 'stars']:
#     if shape_type == 'rectangles':
#         n_shapes = 2
#     else:
#         n_shapes = 4
#     
#     np.random.seed(78727)
#     colors = np.random.dirichlet([.25] * channels, n_colors).reshape([n_colors, channels])
#     bg_colors = np.random.dirichlet([.25] * channels, n_colors).reshape([n_colors, channels])
#     
#     shape_rand = np.random.randint(0, n_shapes, n_images)
#     color_rand = np.random.randint(0, n_colors, n_images)
#     bg_color_rand = np.random.randint(0, n_colors, n_images)
#     # bg_color_rand = (np.random.randint(0, n_colors - 1, n_images) + color_rand + 1) % n_colors
#     patch_offset_v = np.random.randint(0, image_size - patch_size, n_images)
#     patch_offset_h = np.random.randint(0, image_size - patch_size, n_images)
#     
#     plt.figure(figsize=[16, 16])
#     for q in range(16):
#         print q
#         points = base_shape_points(shape_rand[q], random_rotation=random_rotation, mode=shape_type)
#         patch = opacity = convert_points_to_patch(points, patch_size=patch_size, inflation_rate=inflation_rate)
#         colored_patch = patch.reshape([patch_size, patch_size, 1]).dot(colors[color_rand[q]].reshape([1, -1]))
#         
#         image = np.ones([image_size, image_size, 1]).dot(bg_colors[bg_color_rand[q]].reshape([1, -1]))
#         image[patch_offset_v[q]: patch_offset_v[q] + patch_size, patch_offset_h[q]: patch_offset_h[q] + patch_size] *= 1. - np.tile(opacity.reshape([patch_size, patch_size, 1]), [1, 1, channels])
#         image[patch_offset_v[q]: patch_offset_v[q] + patch_size, patch_offset_h[q]: patch_offset_h[q] + patch_size] += colored_patch
#         
#         plt.subplot(4, 4, q + 1)
#         # plt.imshow(patch, interpolation='nearest', cmap=plt.get_cmap('Greys'))
#         fig = plt.imshow(image, interpolation='nearest')
#         plt.title('Shape {}, Color {}, BG {}'.format(shape_rand[q], color_rand[q], bg_color_rand[q]))
#         fig.axes.get_xaxis().set_visible(False)
#         fig.axes.get_yaxis().set_visible(False)
#     
#     plt.savefig('{}.png'.format(shape_type))
#     plt.close()
# 
######################################################################################################################

dirs = ['{}{}/splits/{}/'.format(img_dir, shape_type, ttv) for ttv in ['train', 'test', 'validate']]
for dir in dirs:
    for idx in range(len(dir.split('/')[:-1])):
        if not os.path.exists('/'.join(dir.split('/')[:idx+1])):
            os.mkdir('/'.join(dir.split('/')[:idx+1]))


for shape_type in ['rectangles', 'polygons', 'stars', 'simple'][-1:]:
    print 'Starting {}'.format(shape_type)
    if shape_type == 'simple':
        n_shapes = 2
        random_rotation = False
        random_translation = False
        n_colors = 2
        n_bg_colors = 1
    elif shape_type == 'rectangles':
        n_shapes = 2
        random_rotation = False
        random_translation = True
        n_colors = 4
        n_bg_colors = 4
    else:
        n_shapes = 4
        random_rotation = True
        random_translation = True
        n_colors = 4
        n_bg_colors = 4
    
    print '  Generating random values'
    np.random.seed(78727)
    colors = np.random.dirichlet([.25] * channels, n_colors).reshape([n_colors, channels])
    bg_colors = np.random.dirichlet([.25] * channels, n_bg_colors).reshape([n_bg_colors, channels])
    
    shape_rand = np.random.randint(0, n_shapes, n_images)
    color_rand = np.random.randint(0, n_colors, n_images)
    bg_color_rand = np.random.randint(0, n_bg_colors, n_images)
    # bg_color_rand = (np.random.randint(0, n_bg_colors - 1, n_images) + color_rand + 1) % n_bg_colors
    if random_translation == True:
        patch_offset_v = np.random.randint(50, 50 + image_size - patch_size, n_images)
        patch_offset_h = np.random.randint(25, 25 + image_size - patch_size, n_images)
    else:
        patch_offset_v = np.random.randint(0, 1, size = n_images) + 50 + (image_size - patch_size)/2
        patch_offset_h = np.random.randint(0, 1, size = n_images) + 25 + (image_size - patch_size)/2
    
    print '  Writing feature table'
    feature_table = np.ones([n_images, n_shapes + n_colors + n_bg_colors]) * -1
    for n in xrange(n_images):
        feature_table[n, shape_rand[n]] = 1
        feature_table[n, n_shapes + color_rand[n]] = 1
        feature_table[n, n_shapes + n_colors + bg_color_rand[n]] = 1
    labels = ['{}_{}'.format(shape_type, s) for s in range(n_shapes)] \
             + ['color_{}'.format(c) for c in range(n_colors)] \
             + ['bg_color_{}'.format(c) for c in range(n_bg_colors)]
    attr_file = open('{}{}/list_attr_{}.txt'.format(img_dir, shape_type, shape_type), 'w')
    attr_file.write('{}'.format(n_images) + '\n' + ' '.join(labels) + '\n')
    for n in xrange(n_images):
        attr_file.write('{:06d}.png '.format(n+1) + ' '.join(['{:-2d}'.format(int(val)) for val in feature_table[n, :].tolist()]) + '\n')
    attr_file.close()
    
    print '  Generating and writing images'
    for q in range(n_images):
        points = base_shape_points(shape_rand[q], random_rotation=random_rotation, mode=shape_type)
        patch = opacity = convert_points_to_patch(points, patch_size=patch_size, inflation_rate=inflation_rate)
        colored_patch = patch.reshape([patch_size, patch_size, 1]).dot(colors[color_rand[q]].reshape([1, -1]))
        
        image = np.ones([218, 178, 1]).dot(bg_colors[bg_color_rand[q]].reshape([1, -1]))
        image[patch_offset_v[q]: patch_offset_v[q] + patch_size, patch_offset_h[q]: patch_offset_h[q] + patch_size] *= 1. - np.tile(opacity.reshape([patch_size, patch_size, 1]), [1, 1, channels])
        image[patch_offset_v[q]: patch_offset_v[q] + patch_size, patch_offset_h[q]: patch_offset_h[q] + patch_size] += colored_patch
        image += np.random.normal(0., noise_scale, [218, 178, 3])
        image = np.clip(image, 0., 1.)
        train_test_validate = 'train' if q < n_trn else 'validate' if q < n_trn + n_vld else 'test'
        imsave('{}{}/splits/{}/{:06d}.png'.format(img_dir, shape_type, train_test_validate, q+1), image)
