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

## Usage

### Step 1: extract the data from `*.bag` files

```
cd omnipush-vision
python preprocessing/main.py
```

**Note**: one will probably see error messages by the end of this command, ignore them if the progress bar still forges ahead.
The code will automatically discard damaged bag files.

Aftre running this, one can run `ls data/plywood/1a1a3a2a` and check whether we have generated a `*.h5` file for each `*.bag` file.
Each `*.h5` file contains the following information of a push:

- **RGB_images**: RGB images of a push, (N_RGB, 720, 1280, 3)
- **RGB_time**: Timestamp correspond to RGB images, (N_RGB)
- **depth_images**: Depth images of a push, (N_depth, 720, 1280)
- **depth_time**: Timestamp correspond to depth images, (N_depth)
- **object_pose**: Timestamp and the pose of pushed object, [timestamp, x, y, yaw], (N_op, 4)
- **tip_pose**: Timestamp and the pose of pusher, [timestamp, x, y, yaw], (N_tp, 4)

---

### Step 2: synchronize data from sensors with different frequencies

```
python preprocessing/sync.py
```

Aftre running this, one can run `ls data/plywood/1a1a3a2a` and check whether we have generated a `*_sync.h5` file for each `*.h5` file.
`*_sync.h5` files contain synchrnized information of a push.

---
### Step 3: generate video prediction dataset

```
python video_prediction/gen_dataset.py
```

Now run `ls output` to observe the generated data. Each sub-push should consist of 12 frames with the corresponding action stored in numpy format.

## Questions

- Why the length is shorter (12 frames) than the released dataset?
  - I originally have scripts to stitch different sub-pushes, but they didn't work for `old`, `plywood`, and `old_plywood` :(

