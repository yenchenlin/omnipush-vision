import h5py
import glob
import os
import copy
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle


DATASET_PATH = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/'
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))
OUTPUT_PATH = './dataset'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


dt = 1  # Useless right now since the final frame repeat after 1 second
WORLD_FRAME = True


def rotate(x, theta):
    # x should be 2 dim, [x, y].
    x_rotated = copy.deepcopy(x)
    x_rotated[0] = np.cos(theta) * x[0] + -np.sin(theta) * x[1]
    x_rotated[1] = np.sin(theta) * x[0] + np.cos(theta) * x[1]
    return x_rotated


X = []
Y = []
for fp in tqdm(sync_h5_filepaths):
    src = h5py.File(fp, 'r')
    name = fp.replace('.h5', '').split('/')[-1]
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
        continue

    # Rotate x, y to object's reference frame
    if not WORLD_FRAME:
        delta_object_pose[0:2] = rotate(delta_object_pose[0:2], -theta)

    # Convert radian to degree
    delta_object_pose[2] = delta_object_pose[2] / np.pi * 180.0

    """ Input (tip pose)
    """
    relative_tip_xy = src['tip_pose'][0][1:3] - src['object_pose'][0][1:3]
    if not WORLD_FRAME:
        relative_tip_xy = rotate(relative_tip_xy, -theta)

    tip_vel = (src['tip_pose'][-1][1:3] - src['tip_pose'][0][1:3]) / dt
    if not WORLD_FRAME:
        relative_tip_vel = rotate(tip_vel, -theta)
    else:
        relative_tip_vel = tip_vel
    relative_tip_theta = np.arctan2(relative_tip_vel[1], relative_tip_vel[0])

    relative_tip_pose = np.zeros(3)
    relative_tip_pose[0:2] = relative_tip_xy
    relative_tip_pose[2] = relative_tip_theta / np.pi * 180.0

    """ Create dataset
    """
    dsize = (128, 72)

    x = {}
    x['RGB_image'] = cv2.resize(
        src['RGB_images'][0], dsize=dsize, interpolation=cv2.INTER_CUBIC)
    x['depth_image'] = cv2.resize(
        src['depth_images'][0], dsize=dsize, interpolation=cv2.INTER_CUBIC)
    x['action'] = relative_tip_pose
    x['name'] = name
    X.append(x)

    y = {}
    y['RGB_image'] = cv2.resize(
        src['RGB_images'][-1], dsize=dsize, interpolation=cv2.INTER_CUBIC)
    y['depth_image'] = cv2.resize(
        src['depth_images'][-1], dsize=dsize, interpolation=cv2.INTER_CUBIC)
    y['displacement'] = delta_object_pose
    y['name'] = name
    Y.append(y)


N_train = 8000
X, Y = shuffle(np.array(X), np.array(Y), random_state=0)
X_train = X[:N_train]
Y_train = Y[:N_train]
X_test = X[N_train:]
Y_test = Y[N_train:]
np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_PATH, 'Y_train.npy'), Y_train)
np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_PATH, 'Y_test.npy'), Y_test)
