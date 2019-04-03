#!/usr/bin/env python

# Peter KT Yu, Aug 2015
# parse the bagfiles into simple format with timestamp: probe pose, object pose and ft wrench

import rosbag
import time # for sleep
import json
#from ik.helper import *
from helper import *
from sensor_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import tf
import tf.transformations as tfm
import h5py
import optparse
from cv_bridge import CvBridge, CvBridgeError
import cv2

import sys
import subprocess
import numpy as np

useRRI = True

def ftmsg2listandflip(ftmsg):
    return [-ftmsg.wrench.force.x,ftmsg.wrench.force.y,-ftmsg.wrench.force.z,
            -ftmsg.wrench.torque.x,ftmsg.wrench.torque.y,-ftmsg.wrench.torque.z]
            # the sign flip is for transforming to robot frame

def main(argv):
    global pub
    global pub_ft
    global listener
    global tip_array
    global object_pose_array
    global CartState
    global JointState
    global color_info
    global depth_info
    global color_images
    global depth_images
    global color_times
    global depth_times
    depth_info = None
    tip_array = []
    object_pose_array = []
    CartState = []
    JointState = []
    color_images = []
    depth_images = []
    color_times = []
    depth_times = []

    parser = optparse.OptionParser()
    parser.add_option('', '--json', action="store_true", dest='json',
                      help='To output json or not. ', default=True)
    parser.add_option('', '--nojson', action="store_false", dest='json',
                      help='To output json or not. ', default=True)
    parser.add_option('', '--h5', action="store_true", dest='h5',
                      help='To output h5 or not. ', default=True)
    parser.add_option('', '--noh5', action="store_true", dest='h5',
                      help='To output h5 or not. ', default=True)
    parser.add_option('', '--norri', action="store_false", dest='rri',
                      help='No RRI ', default=True)

    (opt, args) = parser.parse_args()
    if len(args) < 1:  # no bagfile name
        parser.error("Usage: parse_bagfile_to_json.py [bag_file_path.bag]")
        return
    bag_filepath = args[0]
    json_filepath = bag_filepath.replace('.bag', '.json')
    hdf5_filepath = bag_filepath.replace('.bag', '.h5')
    print 'bag_filepath:', bag_filepath
    if opt.json:
        print 'json_filepath:', json_filepath
    if opt.h5:
        print 'hdf5_filepath:', hdf5_filepath
    bridge = CvBridge()



    import rosbag
    bag = rosbag.Bag(bag_filepath)
    #print bag.get_type_and_topic_info().topics
    vicon_to_world = []
    for topic, msg, t in bag.read_messages(topics=['/tf']):
        for i in range(len(msg.transforms)):
            frame_id = msg.transforms[i].header.frame_id
            child_frame_id = msg.transforms[i].child_frame_id
            t = msg.transforms[i].transform
            if child_frame_id == '/viconworld':
                vicon_to_world = [t.translation.x,t.translation.y,t.translation.z] + [t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w]
                break
        if len(vicon_to_world) > 0: break

    if len(vicon_to_world) > 0:
        T_viconworld_to_world = poselist2mat(vicon_to_world)

        for topic, msg, t in bag.read_messages(topics=['/vicon/StainlessSteel/StainlessSteel']):
            pose_viconworld = [msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z,
                msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w]

            T_object_to_viconworld = poselist2mat(pose_viconworld)

            T_object_to_world = np.dot(T_viconworld_to_world, T_object_to_viconworld)

            object_to_world = mat2poselist(T_object_to_world)
            yaw = tfm.euler_from_quaternion(object_to_world[3:7])[2]

            #object_pose_array.append([msg.header.stamp.to_sec()] + object_to_world + [yaw])  # x,y,z, qx,qy,qz,qw, yaw
            object_pose_array.append([msg.header.stamp.to_sec()] + object_to_world[0:2] + [yaw]) # x,y,yaw


    if opt.rri:
        for topic, msg, t in bag.read_messages(topics=['/robot2_RRICartState']):
            rpy = tfm.euler_from_matrix(
            np.dot(tfm.inverse_matrix(tfm.euler_matrix(-np.pi,0,-np.pi)),
            tfm.euler_matrix(msg.position[3]/180.0*np.pi,msg.position[4]/180.0*np.pi,msg.position[5]/180.0*np.pi)))
            # let the preferred pushing orientation be angle zero
            tip_array.append([msg.header.stamp.to_sec(),
                msg.position[0]/1000,msg.position[1]/1000, rpy[2]])  # x,y,theta_yaw

            CartState.append([msg.header.stamp.to_sec(),
                msg.position[0]/1000,msg.position[1]/1000, msg.position[2]/1000, msg.position[3]/180.0*np.pi,msg.position[4]/180.0*np.pi,msg.position[5]/180.0*np.pi])

        for topic, msg, t in bag.read_messages(topics=['/robot2_RRIJointState']):
            JointState.append([msg.header.stamp.to_sec(),
                msg.position[0],msg.position[1], msg.position[2], msg.position[3],msg.position[4],msg.position[5]])
    else:

        for topic, msg, t in bag.read_messages(topics=['/robot2_CartesianLog']):
            rpy = tfm.euler_from_matrix(
            np.dot(tfm.inverse_matrix(tfm.euler_matrix(-np.pi,0,-np.pi)),
            tfm.quaternion_matrix([msg.qx,msg.qy,msg.qz,msg.q0])))

            tip_array.append([msg.timeStamp,
                msg.x/1000,msg.y/1000,rpy[2]])

    for topic, msg, t in bag.read_messages(topics=['/camera/color/camera_info']):
        color_info = [msg.height, msg.width] + np.array(msg.D).tolist() + np.array(msg.K).tolist() + np.array(msg.R).tolist() + np.array(msg.P).tolist()
        break
    for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/camera_info']):
        depth_info = [msg.height, msg.width] + np.array(msg.D).tolist() + np.array(msg.K).tolist() + np.array(msg.R).tolist() + np.array(msg.P).tolist()
        break
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
        color_images.append(cv_image)
        color_times.append(t.to_sec())
    for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
        msg.encoding = "mono16"
        cv_image = bridge.imgmsg_to_cv2(msg, "mono16")
        depth_images.append(cv_image)
        depth_times.append(t.to_sec())

#######
#topics = ['/camera/color/image_raw',  '/tf', '/vicon/StainlessSteel/StainlessSteel', '/robot2_RRICartState',  '/robot2_RRIJointState', '/camera/color/camera_info', '/camera/aligned_depth_to_color/image_raw', '/camera/aligned_depth_to_color/camera_info']

    bag.close()



    # save the data
    if opt.json:
        data = {'tip_pose': tip_array, 'object_pose': object_pose_array,
            'robot_cart': CartState, 'robot_joints': JointState}
            #'RGB_images': color_images, 'depth_images':depth_images,
            #'RGB_time' : color_times, 'depth_time': depth_times}
        with open(json_filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    # save the data as hdf5
    if opt.h5:
        with h5py.File(hdf5_filepath, "w") as f:
            f.create_dataset("tip_pose", data=tip_array)
            f.create_dataset("object_pose", data=object_pose_array)
            f.create_dataset("robot_cart", data=CartState)
            f.create_dataset("robot_joints", data=JointState)
            f.create_dataset("RGB_info", data=color_info)
            f.create_dataset("depth_info", data=depth_info)
            f.create_dataset("RGB_images", data=color_images)
            f.create_dataset("depth_images", data=depth_images)
            f.create_dataset("RGB_time", data=color_times)
            f.create_dataset("depth_time", data=depth_times)


def parse_bagfile(filepath):
    """ Dummy copy of main function, called by `main.py`
    """
    global pub
    global pub_ft
    global listener
    global tip_array
    global object_pose_array
    global CartState
    global JointState
    global color_info
    global depth_info
    global color_images
    global depth_images
    global color_times
    global depth_times
    depth_info = None
    color_info = None
    tip_array = []
    object_pose_array = []
    CartState = []
    JointState = []
    color_images = []
    depth_images = []
    color_times = []
    depth_times = []

    parser = optparse.OptionParser()
    parser.add_option('', '--json', action="store_true", dest='json',
                      help='To output json or not. ', default=True)
    parser.add_option('', '--nojson', action="store_false", dest='json',
                      help='To output json or not. ', default=True)
    parser.add_option('', '--h5', action="store_true", dest='h5',
                      help='To output h5 or not. ', default=True)
    parser.add_option('', '--noh5', action="store_true", dest='h5',
                      help='To output h5 or not. ', default=True)
    parser.add_option('', '--norri', action="store_false", dest='rri',
                      help='No RRI ', default=True)

    (opt, args) = parser.parse_args()
    bag_filepath = filepath
    json_filepath = bag_filepath.replace('.bag', '.json')
    hdf5_filepath = bag_filepath.replace('.bag', '.h5')
    if os.path.exists(hdf5_filepath):
        return
    print 'bag_filepath:', bag_filepath
    # if opt.json:
    #     print 'json_filepath:', json_filepath
    # if opt.h5:
    #     print 'hdf5_filepath:', hdf5_filepath
    bridge = CvBridge()



    import rosbag
    bag = rosbag.Bag(bag_filepath)
    #print bag.get_type_and_topic_info().topics
    vicon_to_world = []
    for topic, msg, t in bag.read_messages(topics=['/tf']):
        for i in range(len(msg.transforms)):
            frame_id = msg.transforms[i].header.frame_id
            child_frame_id = msg.transforms[i].child_frame_id
            t = msg.transforms[i].transform
            if child_frame_id == '/viconworld':
                vicon_to_world = [t.translation.x,t.translation.y,t.translation.z] + [t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w]
                break
        if len(vicon_to_world) > 0: break

    if len(vicon_to_world) > 0:
        T_viconworld_to_world = poselist2mat(vicon_to_world)

        for topic, msg, t in bag.read_messages(topics=['/vicon/StainlessSteel/StainlessSteel']):
            pose_viconworld = [msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z,
                msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w]

            T_object_to_viconworld = poselist2mat(pose_viconworld)

            T_object_to_world = np.dot(T_viconworld_to_world, T_object_to_viconworld)

            object_to_world = mat2poselist(T_object_to_world)
            yaw = tfm.euler_from_quaternion(object_to_world[3:7])[2]

            #object_pose_array.append([msg.header.stamp.to_sec()] + object_to_world + [yaw])  # x,y,z, qx,qy,qz,qw, yaw
            object_pose_array.append([msg.header.stamp.to_sec()] + object_to_world[0:2] + [yaw]) # x,y,yaw


    if opt.rri:
        for topic, msg, t in bag.read_messages(topics=['/robot2_RRICartState']):
            rpy = tfm.euler_from_matrix(
            np.dot(tfm.inverse_matrix(tfm.euler_matrix(-np.pi,0,-np.pi)),
            tfm.euler_matrix(msg.position[3]/180.0*np.pi,msg.position[4]/180.0*np.pi,msg.position[5]/180.0*np.pi)))
            # let the preferred pushing orientation be angle zero
            tip_array.append([msg.header.stamp.to_sec(),
                msg.position[0]/1000,msg.position[1]/1000, rpy[2]])  # x,y,theta_yaw

            CartState.append([msg.header.stamp.to_sec(),
                msg.position[0]/1000,msg.position[1]/1000, msg.position[2]/1000, msg.position[3]/180.0*np.pi,msg.position[4]/180.0*np.pi,msg.position[5]/180.0*np.pi])

        for topic, msg, t in bag.read_messages(topics=['/robot2_RRIJointState']):
            JointState.append([msg.header.stamp.to_sec(),
                msg.position[0],msg.position[1], msg.position[2], msg.position[3],msg.position[4],msg.position[5]])
    else:

        for topic, msg, t in bag.read_messages(topics=['/robot2_CartesianLog']):
            rpy = tfm.euler_from_matrix(
            np.dot(tfm.inverse_matrix(tfm.euler_matrix(-np.pi,0,-np.pi)),
            tfm.quaternion_matrix([msg.qx,msg.qy,msg.qz,msg.q0])))

            tip_array.append([msg.timeStamp,
                msg.x/1000,msg.y/1000,rpy[2]])

    for topic, msg, t in bag.read_messages(topics=['/camera/color/camera_info']):
        color_info = [msg.height, msg.width] + np.array(msg.D).tolist() + np.array(msg.K).tolist() + np.array(msg.R).tolist() + np.array(msg.P).tolist()
        break
    for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/camera_info']):
        depth_info = [msg.height, msg.width] + np.array(msg.D).tolist() + np.array(msg.K).tolist() + np.array(msg.R).tolist() + np.array(msg.P).tolist()
        break
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
        color_images.append(cv_image)
        color_times.append(t.to_sec())
    for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
        msg.encoding = "mono16"
        cv_image = bridge.imgmsg_to_cv2(msg, "mono16")
        depth_images.append(cv_image)
        depth_times.append(t.to_sec())

#######
#topics = ['/camera/color/image_raw',  '/tf', '/vicon/StainlessSteel/StainlessSteel', '/robot2_RRICartState',  '/robot2_RRIJointState', '/camera/color/camera_info', '/camera/aligned_depth_to_color/image_raw', '/camera/aligned_depth_to_color/camera_info']

    bag.close()



    # save the data
    if opt.json:
        data = {'tip_pose': tip_array, 'object_pose': object_pose_array,
            'robot_cart': CartState, 'robot_joints': JointState}
            #'RGB_images': color_images, 'depth_images':depth_images,
            #'RGB_time' : color_times, 'depth_time': depth_times}
        with open(json_filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4)


    # save the data as hdf5
    if opt.h5:
        if not os.path.exists(hdf5_filepath):
            with h5py.File(hdf5_filepath, "w") as f:
                f.create_dataset("tip_pose", data=tip_array)
                f.create_dataset("object_pose", data=object_pose_array)
                f.create_dataset("robot_cart", data=CartState)
                f.create_dataset("robot_joints", data=JointState)
                if color_info is not None:
                    f.create_dataset("RGB_info", data=color_info)
                if depth_info is not None:
                    f.create_dataset("depth_info", data=depth_info)
                f.create_dataset("RGB_images", data=color_images)
                f.create_dataset("depth_images", data=depth_images)
                f.create_dataset("RGB_time", data=color_times)
                f.create_dataset("depth_time", data=depth_times)


if __name__=='__main__':
    main(sys.argv)
