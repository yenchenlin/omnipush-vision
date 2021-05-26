import glob
import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


DATASET_PATH = '/omnipush-vision/data/'
OUTPUT_PATH = '/omnipush-vision/output/'
LENGTH = 12
DSIZE = (64, 64)
N_TEST_PER_SHAPE = 20


np.random.seed(0)
sync_h5_filepaths = sorted(
    glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5')))


def extract_imgs(sync_h5_filepath):
    try:
        sync_h5 = h5py.File(sync_h5_filepath, 'r')
    except:
        return
    shape = sync_h5_filepath.split('/data/')[1].split('/')[0]
    meta = sync_h5_filepath.split('/')[-1].replace('_sync.h5', '')

    dir_train = os.path.join(OUTPUT_PATH, 'train', shape, meta)
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)

    # action's yaw
    try:
        tip_vel = (sync_h5['tip_pose'][-1][1:3] - sync_h5['tip_pose'][0][1:3]) / 1
    except:
        os.rmdir(dir_train)
        return
    tip_theta = np.arctan2(tip_vel[1], tip_vel[0])
    actions = []


    for i in range(LENGTH):
        # image
        try:
            img_cropped = sync_h5['RGB_images'][i][:, 280:1000, :]
        except:
            os.rmdir(dir_train)
            return
        img_resized = cv2.resize(img_cropped,
                                 dsize=DSIZE, interpolation=cv2.INTER_CUBIC)
        img_bgr = img_resized[:, :, ::-1]
        cv2.imwrite(os.path.join(dir_train, '{}.png').format(i), img_bgr)

        # action
        tip_pose = np.zeros(3)
        tip_pose[:2] = sync_h5['tip_pose'][i][1:3]
        tip_pose[2] = tip_theta
        actions.append(tip_pose)
    np.save(os.path.join(dir_train, 'actions.npy'), actions)


Parallel(n_jobs=20)(delayed(extract_imgs)(fp) for fp in tqdm(sync_h5_filepaths))
#for fp in tqdm(sync_h5_filepaths):
#    extract_imgs(fp)
