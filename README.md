# omnipush-vision

## Setup

Clone this repo and put omnipush under `data`:
```
git clone git@github.com:yenchenlin/omnipush-vision.git
mkdir data
mv {/PATH/TO/plywood} data
mv {/PATH/TO/old} data
mv {/PATH/TO/old_plywood} data
```

Then, download and activate the docker image for data preprocessing:
```
docker pull yenchenlin1994/omnipush-vision
docker run -dit -P --name omnipush -v {PATH/TO/omnipush-vision}:/omnipush-vision yenchenlin1994/omnipush-vision
docker attach omnipush
[Press Enter to Continue]
```

where `{PATH/TO/omnipush-vision}` is the path to this repo.

```
cd omnipush-vision
python preprocessing/main.py
```

Aftre running this, we get `path/to/data.h5` files which contain:

- **RGB_images**: RGB images of a push, (N_RGB, 720, 1280, 3)
- **RGB_time**: Timestamp correspond to RGB images, (N_RGB)
- **RGB_info**: ?
- **depth_images**: Depth images of a push, (N_depth, 720, 1280)
- **depth_time**: Timestamp correspond to depth images, (N_depth)
- **depth_info**: ?
- **object_pose**: Timestamp and the pose of pushed object, [timestamp, x, y, yaw], (N_op, 4)
- **tip_pose**: Timestamp and the pose of pusher, [timestamp, x, y, yaw], (N_tp, 4)
- **robot_cart**: Full information of tip_pose, [timestamp, x, y, z, ?, ?, ?], (N_tp, 7)
- **robot_joints**: The rotation of each joint of the robot, (N_tp, 7)

---

Synchronize the data in parallel since different sensors have different frequency:

```
python preprocessing/sync.py
```

Aftre running this, we get `path/to/data_sync.h5` files which trim the amount of samples from every sensor to have the same value as **depth_images** on dimension 0 since **depth_images** has the lowest frequency.

---

To generate dataset for sub-task, run

```
python [TASK]/gen_dataset.py
```

where `[TASK]` can be `video_prediction` | `segmentation`.

## Questions

- What [these lines](https://github.com/yenchenlin/omnipush-vision/blob/master/preprocessing/parse_bagfile_shapes.py#L119-L121) do to get the last dimension of tip_pose? 
