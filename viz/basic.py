#!/usr/bin/env python


import sys
import os, time
import glob
import optparse
import json
import numpy as np
import json
sys.path.append('/home/mcube/pushing_benchmark' + '/Data')
sys.path.append('../helper')
import scipy
import matplotlib.pyplot as plt
import copy
import pdb
import shutil
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
from matplotlib import pyplot as plt
import pdb, copy

# Note: make sure input and output are given in the right dimensions and outputs.
#Parameters
dt = 0.2
input_size = 3
if input_size == 3:
    rotate_inputs = True
else:
    rotate_inputs = False


"""
all_X: [:, 7], where feature means:
'obj_ini_x', 'obj_ini_y', 'obj_ini_angle', 'tip_ini_x', 'tip_ini_y',
'tip_final_x', 'tip_final_y'

all_y: [:, 3], where feature means:
'obj_final_x', 'obj_final_y', 'obj_final_angle'
"""


def process_learning_data(all_X, all_y):
    good_it = []
    aux_X = copy.deepcopy(all_X)
    aux_y = copy.deepcopy(all_y)
    # Velocity of the tip rather than final pose tip
    all_X[:,-2::] = (all_X[:,-2::] - all_X[:,-4:-2])/dt

    """
    all_X: [:, 7], where feature means:
    'obj_ini_x', 'obj_ini_y', 'obj_ini_angle', 'tip_ini_x', 'tip_ini_y',
    'tip_vel_x', 'tip_vel_y'
    """

    # Tranform output
    all_y = all_y - all_X[:,0:3]  # remove initial pose object
    # Ensure delta orientation is correct
    for i in range(all_y.shape[0]):
        if all_y[i, 2] > np.pi: all_y[i, 2] -= 2*np.pi
        if all_y[i, 2] < -np.pi: all_y[i, 2] += 2*np.pi
        if abs(all_y[i, 2]) > np.pi/4:  # Due to Vicon tracker errors
            print('removed', i)
        else:
            good_it.append(i)

    if rotate_inputs:  # only when input has 3 dimensions
        for i in range(all_y.shape[0]):
            theta = all_X[i,2]
            aux_yy = copy.deepcopy(all_y[i,1])
            all_y[i,1] = -np.sin(theta)*all_y[i,0] + np.cos(theta)*all_y[i,1]
            all_y[i,0] = np.cos(theta)*all_y[i,0] + np.sin(theta)*aux_yy

    # Substract initial pose of the object --> only displacement object
    if input_size < 7:
        # Substract initial pose of the object to pose tip
        all_X[:,-4:-2] = all_X[:,-4:-2]-all_X[:,0:2]

    """
    all_X: [:, 7], where feature means:
    'obj_ini_x', 'obj_ini_y', 'obj_ini_angle',
    'tip_ini_x - obj_ini_x', 'tip_ini_y - obj_ini_y',
    'tip_vel_x', 'tip_vel_y'
    """

    if rotate_inputs:  # only when input has 3 dimensions
        #### Rotate all inputs
        for i in range(all_X.shape[0]):
            theta = all_X[i,2]
            aux_yy = copy.deepcopy(all_X[i,4])
            all_X[i,4] = -np.sin(theta)*all_X[i,3] + np.cos(theta)*all_X[i,4]
            all_X[i,3] = np.cos(theta)*all_X[i,3] + np.sin(theta)*aux_yy
            aux_yy = copy.deepcopy(all_X[i,6])
            all_X[i,6] = -np.sin(theta)*all_X[i,5] + np.cos(theta)*all_X[i,6]
            all_X[i,5] = np.cos(theta)*all_X[i,5] + np.sin(theta)*aux_yy
        all_X = all_X[:,3::]

        """
        all_X: [:, 4], where feature means:
        'tip_ini_x - obj_ini_x', 'tip_ini_y - obj_ini_y',
        'tip_vel_x', 'tip_vel_y', all in reference frame
        """

        # Convert velocity to just angle
        all_X[:,-2] = np.arctan2(all_X[:,-1],all_X[:,-2])
        all_X = all_X[:,0:3]
    if input_size == 5:
        all_X = all_X[:,2::]

    all_X = all_X[good_it,:]
    all_y = all_y[good_it,:]

    print('shape: ', all_X.shape)
    return all_X, all_y


#The parameter 'list_filenames' takes a list of files *simple.json
def create_dataset(list_filenames, output_filename):
    list_of_files = glob.glob(list_filenames)
    list_of_files.sort(key=os.path.getctime)
    obj_poses = []
    obj_poses_final = []
    tip_poses = []
    tip_poses_final = []
    for it,file_JSON in enumerate(list_of_files):
        with open(file_JSON) as json_data:
            data = json.load(json_data)
            obj_poses.append(np.array(data['obj_ini'][0:2]+data['obj_ini'][3:4]))
            obj_poses_final.append(np.array(data['obj_end'][0:2]+data['obj_end'][3:4]))
            tip_poses.append(np.array(data['tip_ini'][0:2]))
            tip_poses_final.append(np.array(data['tip_end'][0:2]))
    obj_poses = np.array(obj_poses)
    obj_poses_final = np.array(obj_poses_final)
    tip_poses = np.array(tip_poses)
    tip_poses_final = np.array(tip_poses_final)
    tip = np.concatenate([tip_poses, tip_poses_final], axis = 1)
    io_data = {}
    io_data['input'] = np.concatenate([obj_poses, tip], axis = 1)
    io_data['output'] = obj_poses_final
    np.save(output_filename, io_data)
    return io_data

directory = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/1a1a1a4a/normal'
list_filenames = directory + '/motion_surface=abs_shape=1a1a1a4a_v=50_rep=0046_push=0004_t=-0.452_simple.json'
output_filename = list_filenames.replace('simple.json', '_push_data.npy')
io_data = create_dataset(list_filenames, output_filename)
X, y = process_learning_data(io_data['input'],io_data['output'])
print(X, y)
