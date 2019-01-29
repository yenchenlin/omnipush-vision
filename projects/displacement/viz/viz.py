import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from open3d import *


DATASET_PATH = '/data/vision/phillipi/gen-models/omnipush-vision/projects/displacement/dataset/'
OUTPUT_PATH = '/data/vision/phillipi/gen-models/omnipush-vision/projects/displacement/viz/'
X_PATH = os.path.join(DATASET_PATH, 'X.npy')
Y_PATH = os.path.join(DATASET_PATH, 'Y.npy')
X = np.load(X_PATH)
Y = np.load(Y_PATH)


for i in tqdm(range(X.shape[0])):
    relative_tip_pose = X[i]['action']
    delta_object_pose = Y[i]['displacement']

    """ Process depth image
    """
    rgbd_image_start = create_rgbd_image_from_color_and_depth(
        Image(X[i]['RGB_image']), Image(X[i]['depth_image']))
    rgbd_image_end = create_rgbd_image_from_color_and_depth(
        Image(Y[i]['RGB_image']), Image(Y[i]['depth_image']))

    """ Draw
    """
    fig = plt.figure()
    st = fig.suptitle('Action: {}, \n $\Delta$ obj: {}'.format(
        np.around(relative_tip_pose, decimals=2),
        np.around(delta_object_pose, decimals=2)))
    plt.subplot(2, 2, 1)
    plt.imshow(X[i]['RGB_image'])
    plt.subplot(2, 2, 2)
    plt.imshow(rgbd_image_start.depth)
    plt.subplot(2, 2, 3)
    plt.imshow(Y[i]['RGB_image'])
    plt.subplot(2, 2, 4)
    plt.imshow(rgbd_image_end.depth)
    plt.savefig(os.path.join(OUTPUT_PATH, '{}.png'.format(i)))
