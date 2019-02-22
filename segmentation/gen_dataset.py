import h5py
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


CROP_H = 640
CROP_W = 640
DATASET_PATH = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/'
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')


def extract_imgs(fp):
    src = h5py.File(fp, 'r')
    N_RGB = src['RGB_images'].shape[0]
    timesteps_to_save = [0, 12]
    shape = fp.split('/')[-3]
    metainfo = fp.split('/')[-1].split('_sync.h5')[0]

    for t in timesteps_to_save:
        h, w, _ = src['RGB_images'][t].shape
        start_h = h //2 - (CROP_H // 2)
        start_w = w //2 - (CROP_W // 2)
        img_cropped = src['RGB_images'][t][start_h:start_h+CROP_H, start_w:start_w+CROP_W, :]
        Image.fromarray(img_cropped, 'RGB').save('./dataset/{}_{}_{}.jpg'.format(shape, metainfo, t))


Parallel(n_jobs=20)(delayed(extract_imgs)(fp) for fp in tqdm(sync_h5_filepaths))
