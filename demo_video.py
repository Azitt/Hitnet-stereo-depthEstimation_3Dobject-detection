import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import cv2
import time
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
import time

from utils import get_calibration_parameters, calc_depth_map, find_distances, add_depth,calculate_rms_error, Open3dVisualizer, write_ply,visualize_and_save_3d_boxes,read_disparity
from object_detector import ObjectDetectorAPI

from disparity_estimator.crestereo_disparity_estimator import CREStereoEstimator

from disparity_estimator.hitnet_disparity_estimator import HitNetEstimator
import config

class TimingStats:
    def __init__(self):
        self.total_time = 0
        self.frame_count = 0

    def add_time(self, elapsed_time):
        self.total_time += elapsed_time
        self.frame_count += 1

    def get_average_time(self):
        if self.frame_count == 0:
            return 0
        return self.total_time / self.frame_count

def demo():
    if config.PROFILE_FLAG:
        disp_estimator = None

        if config.ARCHITECTURE == 'crestereo':
            disp_estimator = CREStereoEstimator()
        elif config.ARCHITECTURE == 'hitnet':
            disp_estimator = HitNetEstimator()
        disp_estimator.profile()
        exit()

    left_images = sorted(glob.glob(config.KITTI_LEFT_IMAGES_PATH, recursive=True))
    right_images = sorted(glob.glob(config.KITTI_RIGHT_IMAGES_PATH, recursive=True))
    calib_files = sorted(glob.glob(config.KITTI_CALIB_FILES_PATH, recursive=True))
    gt_disparities = sorted(glob.glob(config.KITTI_disp_PATH, recursive=True)) 
    
    index = 0
    init_open3d = False
    disp_estimator = None
    print("Disparity Architecture Used: {} ".format(config.ARCHITECTURE))

    if config.ARCHITECTURE == 'crestereo':
        disp_estimator = CREStereoEstimator()
    elif config.ARCHITECTURE == 'hitnet':
        disp_estimator = HitNetEstimator()

    
    ### output video ###############################################    
    output_dir = "output_images_withgt_noc0_cre_center6"
    os.makedirs(output_dir, exist_ok=True)  
    # Define the output video file name
    video_filename = f"output_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5  
    frame_duration = 3  
    frame_size = None  

    video_writer = None  
    #error calculator and timing##############################################################
  
    error_accumulator = {'sum_squared_error': 0.0, 'count': 0}
    # Initialize timing stats
    timing_stats = TimingStats()

    for (imfile1, imfile2, calib_file,gt_disparity) in tqdm(list(zip(left_images, right_images, calib_files,gt_disparities))):
        img = cv2.imread(imfile1)
        

        obj_det = ObjectDetectorAPI()
        start = time.time()
        result, pred_bboxes = obj_det.predict(img)
        end = time.time()
        elapsed_time = (end - start) * 1000


        start_d = time.time()
        disparity_map = disp_estimator.estimate(imfile1, imfile2)
        end_d = time.time()
        elapsed_time_d = (end_d - start_d) * 1000
        print("Evaluation Time for Disparity Estimation with {} is : {} ms ".format(config.ARCHITECTURE, elapsed_time_d))
        

        timing_stats.add_time(elapsed_time_d)

        disparity_left = disparity_map
        
        calib_params = get_calibration_parameters(calib_file)
        k_left = calib_params['K_left']
        t_left = calib_params['T_left']
        p_left = calib_params['P_left']
        k_right = calib_params['K_right']
        t_right = calib_params['T_right']
        p_right = calib_params['P_right']
        baseline = calib_params['baseline']
        focal_length = calib_params['focal_length']
        
        # expected baseline=0.54, focallength=1.003556e+3 ###################
        print(baseline,focal_length)
     

        depth_map = calc_depth_map(disparity_map, focal_length, baseline)
          
        if config.ARCHITECTURE == 'hitnet':   
          disparity_map = (disparity_map * 256.).to(torch.uint16) 
          disparity_np = disparity_map.squeeze().cpu().numpy() if isinstance(disparity_map, torch.Tensor) else disparity_map
          color_depth = cv2.applyColorMap(cv2.convertScaleAbs(disparity_np, alpha=0.01), cv2.COLORMAP_JET)
        else:
            disparity_map = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min()) * 255.0
            disp_vis = disparity_map.astype("uint8")
            color_depth = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        
        depth_np = depth_map.detach().cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
        
        depth_list = find_distances(depth_map, pred_bboxes, img, method="center")

        
        ### gt disparity ############################################
        gt_disparity = read_disparity(gt_disparity)
        gt_depth_map = calc_depth_map(gt_disparity, focal_length, baseline) 
        gt_disparity = (gt_disparity * 256.).astype(np.uint16)
        gt_color_depth = cv2.applyColorMap(cv2.convertScaleAbs(gt_disparity, alpha=0.01), cv2.COLORMAP_JET)
        
        gt_depth_list = find_distances(gt_depth_map, pred_bboxes, img, method="center")

        res = add_depth(depth_list, gt_depth_list, result, pred_bboxes,error_accumulator)
       
        
        h, w = img.shape[:2]

        # Ensure that the resized images have even dimensions
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1

        gt_resized = cv2.resize(gt_color_depth, (new_w // 2, new_h // 2))
        pred_resized = cv2.resize(color_depth, (new_w // 2, new_h // 2))

  
        top_part = np.zeros((new_h // 2, new_w, 3), dtype=np.uint8)
        top_part[:, :new_w // 2] = gt_resized
        top_part[:, new_w // 2:] = pred_resized
        

        cv2.putText(top_part, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(top_part, 'Predicted', (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        

        res_resized = cv2.resize(res, (new_w, res.shape[0]))


        combined_image = np.vstack((top_part, res_resized))
        
        if config.SAVE_DISPARITY_OUTPUT:
            
            if frame_size is None:
                frame_size = (combined_image.shape[1], combined_image.shape[0])
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

            for _ in range(frame_duration):
             video_writer.write(combined_image)
            

            ## 3D visualization ############################################### 
            principal_point = (img.shape[1] / 2, img.shape[0] / 2)  # Assuming center of image 
            try:   
             visualization = visualize_and_save_3d_boxes(img, pred_bboxes, np.squeeze(depth_np), focal_length, principal_point)
             if frame_size is None:
                frame_size = (visualization.shape[1], visualization.shape[0])
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

            #  for _ in range(frame_duration):
             video_writer.write(visualization) 
            except Exception as e:
             print(f"Error in visualize_and_save_3d_boxes: {e}") 
                       
        if config.SAVE_DISPARITY_OUTPUT:
            if cv2.waitKey(1) == ord('q'):
                break
    if config.SAVE_DISPARITY_OUTPUT:
        cv2.destroyAllWindows()
    
    
    if video_writer is not None:
        video_writer.release()    
    print(f"Video saved as {video_path}")
    final_rms_error = calculate_rms_error(error_accumulator)
    print(f"Average RMS error over {error_accumulator['count']} measurements: {final_rms_error:.3f} meters")
    average_time = timing_stats.get_average_time()
    print(f"Average execution time for Disparity Estimation with {config.ARCHITECTURE}: {average_time:.2f} ms")
    print(f"Total frames processed: {timing_stats.frame_count}")

if __name__ == '__main__':
    demo()


