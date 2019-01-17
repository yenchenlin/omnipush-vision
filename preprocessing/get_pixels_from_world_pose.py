def get_tranf_matrix(extrinsics, camera_matrix):
    # Matrix for extrinsics
    translate = extrinsics[0:3]
    quaternion = extrinsics[3:7]
    extrinsics_matrix = np.dot(tfm.compose_matrix(translate=translate),tfm.quaternion_matrix(quaternion))[0:3]
   
    # Transformation matrix
    transformation_matrix = np.dot(camera_matrix,extrinsics_matrix)
   
    return transformation_matrix



if __name__=='__main__':
   
    #Default camera info
    aaa = 928.6971435546875; bbb = 657.4966430664062; ccc = 928.1690063476562; ddd = 356.47943115234375
    camera_matrix = np.array([[aaa, 0.0, bbb], [0.0, ccc, ddd], [0.0, 0.0, 1.0]])
    extrinsics = np.array([-0.03229644300182698, -0.020156879828951122, 1.0985830901953102, 0.6398930712215638, 0.5795457241854766, -0.30169529359077646, 0.40452881332006624])
    transformation_matrix = get_tranf_matrix(extrinsics, camera_matrix)
   
    # Project trajectories into pixel space
    vector_des = np.array([x_des, y_des, z_des, z_des*0+1])
    pixels_des = np.dot(transformation_matrix, vector_des)
    pix_x_des = np.divide(pixels_des[0], pixels_des[2])
    pix_y_des = np.divide(pixels_des[1], pixels_des[2])
