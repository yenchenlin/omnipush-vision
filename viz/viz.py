import h5py
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from open3d import *


DATASET_PATH = '/data/vision/phillipi/gen-models/TAKE_THIS_straight_push_all_shapes_no_weight/abs/'
sync_h5_filepaths = glob.glob(os.path.join(DATASET_PATH, '***/**/*_sync.h5'))


src = h5py.File(sync_h5_filepaths[0], 'r')
N_RGB = src['RGB_images'].shape[0]
N_depth = src['depth_images'].shape[0]


rgbd_image = create_rgbd_image_from_color_and_depth(
    Image(src['RGB_images'][0]), Image(src['depth_images'][0]))


plt.subplot(1, 2, 1)
plt.imshow(src['RGB_images'][0])
plt.subplot(1, 2, 2)
plt.imshow(rgbd_image.depth)
plt.savefig('foo.png')


# Show point cloud
pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
draw_geometries([pcd])


# for i in range(N_RGB):
#     Image.fromarray(src['RGB_images'][i], 'RGB').save('{}_RGB.png'.format(i))

# for i in range(N_depth):
#     depth = src['depth_images'][i]
#     np.clip(depth, 0, 2**10 - 1, depth)
#     #depth >>= 2
#     depth = depth.astype(np.uint8)
#     Image.fromarray(depth).save('{}_depth_new.png'.format(i))

#     #cv2.imwrite('{}_depth.png'.format(i), img_scaled)


#     #plt.imshow(src['depth_images'][i], cmap='gray')
#     #plt.savefig('{}_depth.png'.format(i))

#     """
#     depth_img = np.array(src['depth_images'][i], dtype='uint8')
#     cv2.imwrite('{}_depth.png'.format(i), src['depth_images'][i])
#     """
