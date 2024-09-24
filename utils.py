import cv2
import argparse
import glob
import numpy as np
import torch

from PIL import Image
import open3d as o3d
import config
import math

class Open3dVisualizer():
    def __init__(self, K):
        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False
        self.K = K

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
    
    def __call__(self, rgb_image, depth_map, max_dist=20):
        self.update(rgb_image, depth_map, max_dist)

    def update(self, rgb_image, depth_map, max_dist=20):
        # Prepare the rgb image
        rgb_image_resize = cv2.resize(rgb_image, (depth_map.shape[1],depth_map.shape[0]))
        rgb_image_resize = cv2.cvtColor(rgb_image_resize, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image_resize), 
                                                                   o3d.geometry.Image(depth_map),
                                                                   1, depth_trunc=max_dist*1000, 
                                                                   convert_rgb_to_intensity = False)
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.K)
        temp_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # Add values to vectors
        self.point_cloud.points = temp_pcd.points
        self.point_cloud.colors = temp_pcd.colors

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.o3d_started = True

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_front(np.array([ -0.0053112027751292369, 0.28799919460714768, 0.95761592250270977 ]))
            ctr.set_lookat(np.array([-78.783105080589237, -1856.8182240774879, -10539.634663481682]))
            ctr.set_up(np.array([-0.029561736688513099, 0.95716567219818627, -0.28802774118017438]))
            ctr.set_zoom(0.31999999999999978)

        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(config.DEVICE)


def get_calibration_parameters(file):
    parameters = {}
    with open(file, 'r') as f:
        for line in f:
            if line.startswith(('K_02', 'T_02', 'P_rect_02', 'K_03', 'T_03', 'P_rect_03', 'R_02', 'R_03')):
                key, value = line.split(':', 1)
                parameters[key.strip()] = np.array(value.strip().split()).astype(float).reshape(3, -1)
    
    # Calculate baseline
    baseline = abs(parameters['P_rect_03'][0,3] - parameters['P_rect_02'][0,3]) / parameters['P_rect_02'][0,0]
    
    # Focal length (assuming same for both cameras after rectification)
    focal_length = parameters['P_rect_02'][0,0]
    
    return {
        'K_left': parameters['K_02'],
        'T_left': parameters['T_02'],
        'P_left': parameters['P_rect_02'],
        'K_right': parameters['K_03'],
        'T_right': parameters['T_03'],
        'P_right': parameters['P_rect_03'],
        'baseline': baseline,
        'focal_length': focal_length
    }

def find_distances(depth_map, pred_bboxes, img, method="average"):
    depth_list = []
    
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.squeeze().detach().cpu().numpy()
    else:
        depth_map = np.squeeze(depth_map)
    
    depth_h, depth_w = depth_map.shape[:2]
    img_h, img_w, _ = img.shape
    
    # print(f"Depth map shape: {depth_map.shape}")
    # print(f"Image shape: {img.shape}")
    
    # Calculate scaling factors if depths and image have different sizes
    scale_x = depth_w / img_w
    scale_y = depth_h / img_h
    
    for i, box in enumerate(pred_bboxes):
        x1, y1, box_w, box_h, cl, conf = box
        
        # Scale coordinates to match depth map dimensions
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int((x1 + box_w) * scale_x)
        y2_scaled = int((y1 + box_h) * scale_y)
        
        # Ensure coordinates are within depth map bounds
        x1_scaled = max(0, min(x1_scaled, depth_w - 1))
        y1_scaled = max(0, min(y1_scaled, depth_h - 1))
        x2_scaled = max(0, min(x2_scaled, depth_w - 1))
        y2_scaled = max(0, min(y2_scaled, depth_h - 1))
        
        # Check if the box has a valid area
        if x1_scaled >= x2_scaled or y1_scaled >= y2_scaled:
            print(f"Box {i}: Invalid dimensions. Skipping.")
            depth_list.append(np.nan)
            continue
        
        obstacle_depth = depth_map[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
        
        if obstacle_depth.size == 0:
            print(f"Box {i}: No valid depth data. Skipping.")
            depth_list.append(np.nan)
            continue
        
        if method == "closest":
            depth = np.nanmin(obstacle_depth)
        elif method == "average":
            depth = np.nanmean(obstacle_depth)
        elif method == "median":
            depth = np.nanmedian(obstacle_depth)
        else:  # center
            center_y = (y1_scaled + y2_scaled) // 2
            center_x = (x1_scaled + x2_scaled) // 2
            depth = depth_map[center_y, center_x]
        

        depth_list.append(depth)
    
    return depth_list

def compare_depths(pred_depth_map, gt_depth_map, pred_bboxes, img, methods=["center", "average"]):
    results = {}
    for method in methods:
        pred_depths = find_distances(pred_depth_map, pred_bboxes, img, method=method)
        gt_depths = find_distances(gt_depth_map, pred_bboxes, img, method=method)
        
        # Remove any NaN values
        valid_indices = ~(np.isnan(pred_depths) | np.isnan(gt_depths))
        pred_depths = np.array(pred_depths)[valid_indices]
        gt_depths = np.array(gt_depths)[valid_indices]
        
        # Calculate metrics
        abs_rel_error = np.mean(np.abs(gt_depths - pred_depths) / gt_depths)
        sq_rel_error = np.mean(((gt_depths - pred_depths) ** 2) / gt_depths)
        rmse = np.sqrt(np.mean((gt_depths - pred_depths) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(gt_depths) - np.log(pred_depths)) ** 2))
        
        thresh = np.maximum((gt_depths / pred_depths), (pred_depths / gt_depths))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        
        results[method] = {
            "Abs Rel": abs_rel_error,
            "Sq Rel": sq_rel_error,
            "RMSE": rmse,
            "RMSE log": rmse_log,
            "δ < 1.25": a1,
            "δ < 1.25²": a2,
            "δ < 1.25³": a3
        }
    
    return results 

def read_disparity(disparity_path):
    """Read and preprocess the disparity map."""
    if isinstance(disparity_path, str):
        # Read the disparity map from file
        disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise FileNotFoundError(f"Could not read disparity file: {disparity_path}")
        
        # KITTI stores disparity as uint16, scale factor is 256
        disparity = disparity.astype(np.float32) / 256.0
    elif isinstance(disparity_path, np.ndarray):
        disparity = disparity_path
    else:
        raise TypeError("Input must be either a file path or a numpy array")
    
    return disparity 


def calc_depth_map(disp_left, focal_length, baseline):
    """
    Calculate a depth map from disparity map.
    
    Args:
    disp_left (torch.Tensor): Disparity map
    focal_length (float): Focal length of the camera
    baseline (float): Baseline distance between stereo cameras
    
    Returns:
    torch.Tensor: Depth map
    """
    # Create a mask for valid disparity values
    mask = (disp_left > 0) & (disp_left < 1000)  # Adjust the upper bound as needed
    
    # Initialize depth map
    if isinstance (disp_left,torch.Tensor):
      depth_map = torch.zeros_like(disp_left)
    else:
      depth_map = np.zeros_like(disp_left)    
    
    # Calculate depth only for valid disparities
    depth_map[mask] = (focal_length * baseline) / disp_left[mask]
    
    # Clip depth to a reasonable range (e.g., 1m to 100m)
    if isinstance (depth_map,torch.Tensor):
       depth_map = torch.clamp(depth_map, 1.0, 100.0)
    else:
       depth_map = np.clip(depth_map, 1.0, 100.0) 
    
    return depth_map

def draw_depth(depth_map, img_width, img_height, max_dist=10):
		
	return util_draw_depth(depth_map, (img_width, img_height), max_dist)

def draw_disparity(disparity_map, img_width, img_height):

    disparity_map =  cv2.resize(disparity_map,  (img_width, img_height))
    norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
                              (np.max(disparity_map)-np.min(disparity_map)))

    
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

def add_depth(depth_list, gt_depth_list, result, pred_bboxes,error_accumulator):
    h, w, _ = result.shape
    res = result.copy()

    for i, (box, distance, gt_distance) in enumerate(zip(pred_bboxes, depth_list, gt_depth_list)):
        x1, y1, w, h, _, _ = map(int, box[:6])
        x2, y2 = x1 + w, y1 + h
                
        # Prepare depth texts
        depth_text = f'Pred: {distance:.2f}m'
        gt_depth_text = f'GT: {gt_distance:.2f}m'
        
        # Calculate text sizes
        depth_text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        gt_depth_text_size = cv2.getTextSize(gt_depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Calculate positions
        text_x = x1 + (x2 - x1 - max(depth_text_size[0], gt_depth_text_size[0])) // 2
        gt_text_y = y1 + (y2 - y1) // 2 - 5  # 5 pixels above the center
        pred_text_y = y1 + (y2 - y1) // 2 + gt_depth_text_size[1] + 5  # 5 pixels below the center
        
        if 1.0 < gt_distance < 5.0:
            # Accumulate squared error for valid range
            error_accumulator['sum_squared_error'] += (distance - gt_distance) ** 2
            error_accumulator['count'] += 1
            
        if gt_distance > 1.0:
         # Put ground truth depth text (in green)
         cv2.putText(res, gt_depth_text, (text_x, gt_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Put predicted depth text (in red)
        cv2.putText(res, depth_text, (text_x, pred_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return res

def calculate_rms_error(error_accumulator):
    if error_accumulator['count'] == 0:
        return 0.0
    
    mean_squared_error = error_accumulator['sum_squared_error'] / error_accumulator['count']
    rms_error = math.sqrt(mean_squared_error)
    return rms_error
## create 3D boxes #################################################

def draw_cuboid(img, corners, color=(0, 255, 0), thickness=2):
    def draw_line(p1, p2):
        cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), color, thickness)

    # Draw front face
    for i in range(4):
        draw_line(corners[i], corners[(i+1) % 4])

    # Draw back face
    for i in range(4):
        draw_line(corners[i+4], corners[((i+1) % 4) + 4])

    # Draw connecting lines
    for i in range(4):
        draw_line(corners[i], corners[i+4])

def visualize_and_save_3d_boxes(img, pred_bboxes, depth_map, focal_length, principal_point):
    
    
    visualization = img.copy()
    h, w = img.shape[:2]

    for bbox in pred_bboxes:
        x, y, width, height, _, _ = map(int, bbox[:6])
        depth = depth_map[y + height//2, x + width//2]
    
        # 3D box dimensions (in world coordinates)
        box_width = width * depth / focal_length
        box_height = height * depth / focal_length
        box_depth = box_width * 1.5  # Adjust this multiplier as needed
        
        # 3D box center
        center_x = (x + width/2 - principal_point[0]) * depth / focal_length
        center_y = (y + height/2 - principal_point[1]) * depth / focal_length
        center_z = depth
        
        # Define 3D box corners
        corners_3d = np.array([
            [center_x - box_width/2, center_y - box_height/2, center_z - box_depth/2],
            [center_x + box_width/2, center_y - box_height/2, center_z - box_depth/2],
            [center_x + box_width/2, center_y + box_height/2, center_z - box_depth/2],
            [center_x - box_width/2, center_y + box_height/2, center_z - box_depth/2],
            [center_x - box_width/2, center_y - box_height/2, center_z + box_depth/2],
            [center_x + box_width/2, center_y - box_height/2, center_z + box_depth/2],
            [center_x + box_width/2, center_y + box_height/2, center_z + box_depth/2],
            [center_x - box_width/2, center_y + box_height/2, center_z + box_depth/2],
        ])

        # Project 3D points to 2D
        corners_2d = []
        for corner in corners_3d:
            px = int(corner[0] * focal_length / corner[2] + principal_point[0])
            py = int(corner[1] * focal_length / corner[2] + principal_point[1])
            corners_2d.append([px, py])

        # Draw cuboid
        color = (0, 255, 0) #(0, 255 - int(depth * 5), 0)  # Adjusted color scaling
        thickness = 1 #max(1, int(4 - depth/10))  # Adjusted thickness scaling
        draw_cuboid(visualization, corners_2d, color, thickness)
        
    return visualization


##################################################################################################
def util_draw_depth(depth_map, img_shape, max_dist):

	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] = 0
	norm_depth_map[norm_depth_map >= 255] = 0
	norm_depth_map =  cv2.resize(norm_depth_map, img_shape)
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'ab') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
