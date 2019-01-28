import h5py
import glob
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from open3d import *
from tqdm import tqdm


DATASET_PATH = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/'
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))
CHECK_PATH = './check'
if not os.path.exists(CHECK_PATH):
    os.mkdir(CHECK_PATH)


# TODO: modify later
dt = 1


def rotate(x, theta):
    # x should be 2 dim, [x, y].
    x_rotated = copy.deepcopy(x)
    x_rotated[0] = np.cos(theta) * x[0] + -np.sin(theta) * x[1]
    x_rotated[1] = np.sin(theta) * x[0] + np.cos(theta) * x[1]
    return x_rotated


for fp in tqdm(sync_h5_filepaths):
    src = h5py.File(fp, 'r')
    tar = fp.replace('.h5', '.png').split('/')[-1]
    theta = src['object_pose'][0][3]

    """ Output (displacement)
    """
    delta_object_pose = src['object_pose'][-1][1:] - src['object_pose'][0][1:]

    # Handle theta when out of range
    if delta_object_pose[2] > np.pi:
        delta_object_pose -= 2 * np.pi
    if delta_object_pose[2] < -np.pi:
        delta_object_pose += 2 * np.pi
    if abs(delta_object_pose[2]) > np.pi/4:  # Due to Vicon tracker errors
        print("Vicon tracker errors")
        exit()

    # Rotate x, y to object's reference frame
    delta_object_pose[0:2] = rotate(delta_object_pose[0:2], -theta)

    # Convert radian to degree
    delta_object_pose[2] = delta_object_pose[2] / np.pi * 180.0

    """ Input (tip pose)
    """
    relative_tip_xy = src['tip_pose'][0][1:3] - src['object_pose'][0][1:3]
    relative_tip_xy = rotate(relative_tip_xy, -theta)

    tip_vel = (src['tip_pose'][-1][1:3] - src['tip_pose'][0][1:3]) / dt
    #relative_tip_vel = rotate(tip_vel, -theta)
    relative_tip_vel = tip_vel
    relative_tip_theta = np.arctan2(relative_tip_vel[1], relative_tip_vel[0])

    relative_tip_pose = np.zeros(3)
    relative_tip_pose[0:2] = relative_tip_xy
    relative_tip_pose[2] = relative_tip_theta / np.pi * 180.0

    """ Process depth image
    """
    # rgbd_image_start = create_rgbd_image_from_color_and_depth(
    #     Image(src['RGB_images'][0]), Image(src['depth_images'][0]))
    # rgbd_image_end = create_rgbd_image_from_color_and_depth(
    #     Image(src['RGB_images'][-1]), Image(src['depth_images'][-1]))

    """ Draw
    """
    DIR = os.path.join(CHECK_PATH, fp.replace('.h5', '').split('/')[-1])
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    for i in range(src['RGB_images'].shape[0]):
        fig = plt.figure()
        st = fig.suptitle('$\Delta$ obj: {},\n Rel tip: {}'.format(
            np.around(delta_object_pose, decimals=2),
            np.around(relative_tip_pose, decimals=2)))
        plt.imshow(src['RGB_images'][i])
        plt.savefig(os.path.join(DIR, '{}.png'.format(i)))
        plt.close()
    # fig = plt.figure()
    # st = fig.suptitle('$\Delta$ obj: {},\n Rel tip: {}'.format(
    #     np.around(delta_object_pose, decimals=2),
    #     np.around(relative_tip_pose, decimals=2)))
    # plt.subplot(2, 2, 1)
    # plt.imshow(src['RGB_images'][0])
    # plt.subplot(2, 2, 2)
    # plt.imshow(rgbd_image_start.depth)
    # plt.subplot(2, 2, 3)
    # plt.imshow(src['RGB_images'][-1])
    # plt.subplot(2, 2, 4)
    # plt.imshow(rgbd_image_end.depth)
    # plt.savefig(os.path.join(CHECK_PATH, tar))
