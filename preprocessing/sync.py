import os
import glob
import h5py
from bisect import bisect_left
from joblib import Parallel, delayed
from tqdm import tqdm


FIELDS = ['RGB_time', 'object_pose', 'tip_pose', 'robot_cart'] # 'robot_joints']


def argclosest(input_list, input_num):
    """
    Assumes input_list is sorted. Returns closest value to input_num.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(input_list, input_num)
    if pos == 0:
        return 0
    if pos == len(input_list):
        return len(input_list)-1
    before = input_list[pos - 1]
    after = input_list[pos]
    if abs(after - input_num) < abs(input_num - before):
        return pos
    else:
        return pos-1


def sync_data(h5_filepath):
    src = h5py.File(h5_filepath, 'r')
    time = {}
    time['depth_time'] = src['depth_time']
    time['RGB_time'] = src['RGB_time']
    try:
        time['object_pose'] = src['object_pose'][:, 0]
    except:
        print('Skipping {} due to uncomplete data.'.format(h5_filepath))
        return
    time['tip_pose'] = src['tip_pose'][:, 0]
    time['robot_cart'] = src['robot_cart'][:, 0]
    # time['robot_joints'] = src['robot_joints'][:, 0]

     # Make sure depth has least values along time
    for field in FIELDS:
        assert time['depth_time'].shape[0] <= time[field].shape[0] + 1
        # Add 1 because one push's # of RGB images == # of depth images - 1

    data = {}
    data['RGB_images'] = []  # It is copied when dealing with 'RGB_time'
    for field in FIELDS:
        data[field] = []

    # Trim the dimension of each sensor data to the dimension of
    # depth camera since it has the lowest frequency
    for field in FIELDS:
        for dt in src['depth_time']:
            nn = argclosest(time[field], dt)
            data[field].append(src[field][nn])
            # Handle 'RGB_images' when dealing with 'RGB_time'
            if field == 'RGB_time':
                data['RGB_images'].append(src['RGB_images'][nn])

    sync_h5_filepath = h5_filepath.replace('.h5', '_sync.h5')
    with h5py.File(sync_h5_filepath, "w") as f:
        # Copy source depth image to sync data
        f.create_dataset('depth_images', data=src['depth_images'])
        f.create_dataset('depth_time', data=src['depth_time'])
        # Copy trim data to sync data
        f.create_dataset('tip_pose', data=data['tip_pose'])
        f.create_dataset('object_pose', data=data['object_pose'])
        f.create_dataset('robot_cart', data=data['robot_cart'])
        # f.create_dataset('robot_joints', data=data['robot_joints'])
        f.create_dataset('RGB_images', data=data['RGB_images'])
        f.create_dataset('RGB_time', data=data['RGB_time'])


DATASET_PATH = '/omnipush-vision/data/'

# Get .h5 filepaths except _sync.h5 filepaths
h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*.h5'))
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))
h5_filepaths = [fp for fp in h5_filepaths if fp not in sync_h5_filepaths]

# Get un-sync .h5 filepaths
sync_h5_filepaths = [fp.replace('.h5', '_sync.h5') for fp in h5_filepaths]
unfinished_h5_filepaths = []
for sync_fp, fp in zip(sync_h5_filepaths, h5_filepaths):
    if not os.path.exists(sync_fp):
        unfinished_h5_filepaths.append(fp)

print("{}/{} h5 files un-sync.".format(len(unfinished_h5_filepaths), len(h5_filepaths)))

for fp in tqdm(unfinished_h5_filepaths):
     try:
         sync_data(fp)
     except IOError:
         print("Bad file: {}".format(fp))
# Parallel(n_jobs=20)(delayed(sync_data)(fp) for fp in tqdm(unfinished_h5_filepaths))
