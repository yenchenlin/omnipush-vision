import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from parse_bagfile_shapes import parse_bagfile

DATASET_PATH = '/omnipush-vision/data/'
bag_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*.bag'))
Parallel(n_jobs=20)(delayed(parse_bagfile)(fp) for fp in tqdm(bag_filepaths))
