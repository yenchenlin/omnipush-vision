import os
import glob
import h5py
import numpy as np
from tqdm import tqdm


DATASET_PATH = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/'
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))


stats = {}
stats['object_pose'] = {}
stats['object_pose']['start'] = np.empty([len(sync_h5_filepaths), 4])
stats['object_pose']['end'] = np.empty([len(sync_h5_filepaths), 4])


for i, sync_h5_filepath in enumerate(tqdm(sync_h5_filepaths)):
    src = h5py.File(sync_h5_filepath, 'r')
    stats['object_pose']['start'][i] = src['object_pose'][0]
    stats['object_pose']['end'][i] = src['object_pose'][-1]


np.save('object_pose', stats['object_pose'])
