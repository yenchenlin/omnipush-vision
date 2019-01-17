# omnipush-vision

## Usage

Preprocess `path/to/data.bag` files in parallel:

```
python preprocessing/main.py
```

Aftre running this, we get `path/to/data.h5` files which contain:

- RGB_images: RGB images of a push, (N_RGB, 720, 1280, 3)
- RGB_time: Timestamp correspond to RGB images, (N_RGB)
- RGB_info: ?
- depth_images: Depth images of a push, (N_depth, 720, 1280)
- depth_time: Timestamp correspond to depth images, (N_depth)
- depth_info: ?
- object_pose: Timestamp and the pose of pushed object, [timestamp, x, y, yaw], (N_op, 4)
- tip_pose: Timestamp and the pose of pusher, [timestamp, x, y, yaw], (N_tp, 4)
- robot_cart: Full information of tip_pose, [timestamp, x, y, z, ?, ?, ?], (N_tp, 7)
- robot_joints: The rotation of each joint of the robot, (N_tp, 7)

## Questions

- What [these lines](https://github.com/yenchenlin/omnipush-vision/blob/master/preprocessing/parse_bagfile_shapes.py#L119-L121) do to get the last dimension of tip_pose? 