import torch

####### reading test images #############################
# KITTI_CALIB_FILES_PATH="./data/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
# KITTI_LEFT_IMAGES_PATH="./data/data_scene_flow/testing/image_2/*.png"
# KITTI_RIGHT_IMAGES_PATH="./data/data_scene_flow/testing/image_3/*.png"
## reading from training ################################
KITTI_CALIB_FILES_PATH="./data/data_scene_flow_calib/training/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH="./data/data_scene_flow/training/image_2/*_10.png"
KITTI_RIGHT_IMAGES_PATH="./data/data_scene_flow/training/image_3/*_10.png"
KITTI_disp_PATH="./data/data_scene_flow/training/disp_noc_0/*.png"

#########################################################

CRESTEREO_MODEL_PATH = "pretrained_models/crestereo/crestereo_eth3d.pth"
HITNET_MODEL_PATH = "pretrained_models/hitnet/bestD1_checkpoint.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE1 = "cuda:0" if torch.cuda.is_available() else "cpu"



ARCHITECTURE_LIST = ['crestereo', 'hitnet']
ARCHITECTURE = ARCHITECTURE_LIST[0]
SAVE_POINT_CLOUD = 0
SAVE_DISPARITY_OUTPUT = 1
SHOW_3D_PROJECTION = 0

# Enable this Profile flag only when you want to Profile
PROFILE_FLAG = 0
PROFILE_IMAGE_WIDTH = 512
PROFILE_IMAGE_HEIGHT = 512


