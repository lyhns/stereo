import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 确保负号正确显示
#plt.rcParams['axes.unicode_minus'] = False

class DisparityToDepth:
    def __init__(self,focal_length= 450.0):
        self.baseline = None
        self.focal_length = focal_length  # Focal length in pixels
        self.depth = None
        self.disparity = None  # 存储视差图用于可视化
        self.camera_poses = {}
        self.prev_depth = None
        self.prev_depth_norm = None

    def load_camera_poses_from_file(self, file_path):
    #Load camera pose data from TXT file
        self.camera_poses = {}
        current_frame = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split()
                if parts[0] == 'Frame':
                    # Parse frame number
                    current_frame = int(parts[1])
                    self.camera_poses[current_frame] = {}
                elif parts[0] in ['L', 'R'] and current_frame is not None:
                    # Parse left/right camera pose data
                    cam = parts[0]
                    pose_values = list(map(float, parts[1:]))
                    self.camera_poses[current_frame][cam] = np.array(pose_values).reshape(4, 4)
        
        # 计算基线：从第一个同时包含左右相机位姿的帧中提取
        self.baseline = None
        # 获取所有包含左右相机位姿的有效帧
        valid_frames = [f for f in self.camera_poses 
                       if 'L' in self.camera_poses[f] and 'R' in self.camera_poses[f]]
    
        if valid_frames:
            # 选择第一个有效帧计算基线
            first_valid_frame = valid_frames[0]
            t_l = self.camera_poses[first_valid_frame]['L'][:3, 3]
            t_r = self.camera_poses[first_valid_frame]['R'][:3, 3]
            self.baseline = np.linalg.norm(t_r - t_l)
            
            print(f"Camera poses parsed successfully:")
            print(f"  Total valid frames: {len(valid_frames)}")
            print(f"  Calculated baseline from frame {first_valid_frame}")
            print(f"  Baseline length: {self.baseline:.4f}m")
            print(f"  Focal length: {self.focal_length:.1f}pixels")
        else:
            print("Warning: No valid frames with both left and right camera poses found")
            self.baseline = None  # 未找到有效帧时基线设为None

    def read_image(self, image_path):
        """Read image from file (supports common formats)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    def read_disparity(self, disparity_path=None, disparity_array=None):
        if disparity_array is not None:
            self.disparity = disparity_array.astype(np.float32)
            return self.disparity

        if disparity_path.endswith('.pfm'):
            self.disparity = self._read_pfm(disparity_path)
        elif disparity_path.endswith('.npy'):
            self.disparity = np.load(disparity_path).astype(np.float32)
        elif disparity_path.endswith(('.png', '.jpg')):
            self.disparity = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        else:
            raise ValueError("Only supports .pfm, .npy or image format disparity files")
        
        return self.disparity

    def _read_pfm(self, file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header != 'Pf':
                raise ValueError("Only supports single-channel PFm disparity files")

            width, height = map(int, f.readline().split())
            scale = float(f.readline().rstrip())
            endian = '<' if scale < 0 else '>'
            scale_abs = abs(scale)

            disp_data = np.fromfile(f, dtype=endian + 'f')
            disparity = disp_data.reshape((height, width))[::-1]
            return disparity * scale_abs

    def convert(self, disparity_path=None, disparity_array=None, frame_id=400, min_disparity=1):
        if frame_id not in self.camera_poses:
            raise ValueError(f"Frame {frame_id} camera pose data not found")

        if self.baseline is None:
            t_l = self.camera_poses[frame_id]['L'][:3, 3]
            t_r = self.camera_poses[frame_id]['R'][:3, 3]
            self.baseline = np.linalg.norm(t_r - t_l)

        # 读取视差图（会存储在self.disparity中）
        disparity = self.read_disparity(disparity_path, disparity_array)

        # 过滤无效视差值
        mask = (disparity < min_disparity) | np.isnan(disparity) | np.isinf(disparity)
        disparity[mask] = np.nan

        # 计算深度
        self.depth = (self.baseline * self.focal_length) / disparity
        return self.depth

    def visualize_stereo_disparity_depth(self, frame_id=400, cmap_depth='viridis', cmap_disp='jet',
                                        left_image_path=None, right_image_path=None,
                                        save_fig=False, save_dir='stereo_visualizations'):
        """
        可视化布局：
        - 上方：左视图（左）和右视图（右）
        - 下方：视差图（左）和深度图（右）
        """
        if self.depth is None or self.disparity is None:
            raise ValueError("Please call convert() first to generate depth and disparity maps")

        # 准备深度图可视化数据
        depth_vis = self.depth.copy()
        invalid_mask_depth = np.isnan(depth_vis) | np.isinf(depth_vis)
        valid_depth = depth_vis[~invalid_mask_depth]
        has_valid_depth = len(valid_depth) > 0

        # 处理深度图无效值
        if has_valid_depth:
            valid_max_depth = np.max(valid_depth)
            depth_vis[invalid_mask_depth] = valid_max_depth
            min_depth = np.min(depth_vis)
            max_depth = np.max(depth_vis)
            depth_range_text = f"Depth range: {min_depth:.2f}-{max_depth:.2f}m"
            
            if max_depth == min_depth:
                depth_norm = np.zeros_like(depth_vis, dtype=np.uint8)
            else:
                depth_norm = (depth_vis - min_depth) / (max_depth - min_depth) * 255
                depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_vis, dtype=np.uint8)
            depth_range_text = "No valid depth values"

        # 准备视差图可视化数据
        disp_vis = self.disparity.copy()
        invalid_mask_disp = np.isnan(disp_vis) | np.isinf(disp_vis) | (disp_vis <= 0)
        valid_disp = disp_vis[~invalid_mask_disp]
        has_valid_disp = len(valid_disp) > 0

        # 处理视差图无效值
        if has_valid_disp:
            valid_max_disp = np.max(valid_disp)
            disp_vis[invalid_mask_disp] = valid_max_disp
            min_disp = np.min(disp_vis)
            max_disp = np.max(disp_vis)
            disp_range_text = f"Disparity range: {min_disp:.2f}-{max_disp:.2f}px"
            
            if max_disp == min_disp:
                disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
            else:
                disp_norm = (disp_vis - min_disp) / (max_disp - min_disp) * 255
                disp_norm = np.clip(disp_norm, 0, 255).astype(np.uint8)
        else:
            disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
            disp_range_text = "No valid disparity values"

        # 创建保存目录
        if save_fig and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 创建2x2布局的图像
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f"Stereo Vision Data - Frame {frame_id}", fontsize=20)

        # 上方左：左视图
        if left_image_path and os.path.exists(left_image_path):
            try:
                left_img = self.read_image(left_image_path)
                axes[0, 0].imshow(left_img)
                axes[0, 0].set_title('Left Camera View')
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f"Left image\nnot available", 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, f"Left image\npath not provided", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 0].transAxes)
        axes[0, 0].axis('off')

        # 上方右：右视图
        if right_image_path and os.path.exists(right_image_path):
            try:
                right_img = self.read_image(right_image_path)
                axes[0, 1].imshow(right_img)
                axes[0, 1].set_title('Right Camera View')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f"Right image\nnot available", 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, f"Right image\npath not provided", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')

        # 下方左：视差图
        im_disp = axes[1, 0].imshow(disp_norm, cmap=cmap_disp)
        axes[1, 0].set_title('Disparity Map')
        axes[1, 0].axis('off')
        # 视差图颜色条
        #cbar_disp = fig.colorbar(im_disp, ax=axes[1, 0], fraction=0.046, pad=0.04)
        #cbar_disp.set_label(disp_range_text)

        disp_bbox = axes[1, 0].get_position()  # 获取视差图的边界框（[左, 下, 宽, 高]）
        cbar_width = 0.015  # 颜色条宽度（可微调）
        cbar_left = disp_bbox.x1 + 0.005  # 颜色条左侧位置（视差图右侧+微小间距）
        cbar_bottom = disp_bbox.y0  # 颜色条底部与视差图底部对齐
        cbar_height = disp_bbox.height  # 颜色条高度与视差图高度一致

        # 添加颜色条（使用精确坐标定位）
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar_disp = fig.colorbar(im_disp, cax=cbar_ax)
        cbar_disp.set_label(disp_range_text, rotation=270, labelpad=15)  # 标签垂直显示
        

        # 下方右：深度图
        im_depth = axes[1, 1].imshow(depth_norm, cmap=cmap_depth)
        axes[1, 1].set_title('Depth Map')
        axes[1, 1].axis('off')
        # 深度图颜色条
        #cbar_depth = fig.colorbar(im_depth, ax=axes[1, 1], fraction=0.046, pad=0.04)
        #cbar_depth.set_label(depth_range_text)

        disp_bbox = axes[1, 1].get_position()  # 获取视差图的边界框（[左, 下, 宽, 高]）
        cbar_width = 0.015  # 颜色条宽度（可微调）
        cbar_left = disp_bbox.x1 + 0.005  # 颜色条左侧位置（视差图右侧+微小间距）
        cbar_bottom = disp_bbox.y0  # 颜色条底部与视差图底部对齐
        cbar_height = disp_bbox.height  # 颜色条高度与视差图高度一致

        # 添加颜色条（使用精确坐标定位）
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar_disp = fig.colorbar(im_depth, cax=cbar_ax)
        cbar_disp.set_label(depth_range_text, rotation=270, labelpad=15)  # 标签垂直显示
                     
        # 调整布局
        #plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间

        # 保存图像
        if save_fig:
            filename = f'stereo_disparity_depth_frame_{frame_id}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def save_depth(self, save_path, format='npy'):
        if self.depth is None:
            raise ValueError("Please call convert() first to generate depth map")

        if format == 'npy':
            np.save(save_path, self.depth)
            print(f"Depth data saved to {save_path}")
        elif format == 'png':
            depth_vis = self.depth.copy()
            invalid_mask = np.isnan(depth_vis) | np.isinf(depth_vis)
            valid_depth = depth_vis[~invalid_mask]

            if len(valid_depth) == 0:
                depth_norm = np.zeros_like(depth_vis, dtype=np.uint8)
            else:
                valid_max = np.max(valid_depth)
                depth_vis[invalid_mask] = valid_max
                min_depth = np.min(depth_vis)
                max_depth = np.max(depth_vis)
                if max_depth == min_depth:
                    depth_norm = np.zeros_like(depth_vis, dtype=np.uint8)
                else:
                    depth_norm = (depth_vis - min_depth) / (max_depth - min_depth) * 255
                    depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, depth_norm)
            print(f"Depth image saved to {save_path}")
        else:
            raise ValueError("Only supports npy or png format")


# 使用示例
if __name__ == "__main__":
    # 初始化转换器
    converter = DisparityToDepth(focal_length= 450.0)
    
    # 从TXT文件加载相机位姿数据
    base_path = "/root/pfm/"
    pose_file_path = base_path + "camera_data.txt"
    print(f"Loading camera poses from {pose_file_path}...")
    converter.load_camera_poses_from_file(pose_file_path)
    
    # 处理第400帧
    frame_id = 400
    frame_id_str = str(400)
    disparity_path = base_path + frame_id_str+ ".pfm"
    left_img_path = base_path + "rgb/left/" + frame_id_str+".png"   # 替换为实际左视图路径
    right_img_path = base_path + "rgb/right/" + frame_id_str +".png" # 替换为实际右视图路径
    
    print(f"\nProcessing frame {frame_id}...")
    converter.convert(disparity_path, frame_id=frame_id)
    
    # 可视化四视图布局
    converter.visualize_stereo_disparity_depth(
        frame_id=frame_id,
        left_image_path=left_img_path,
        right_image_path=right_img_path,
        save_fig=True
    )
    
    # 保存深度数据
    converter.save_depth(f"depth_frame_{frame_id}.npy")
    converter.save_depth(f"depth_visualization_frame_{frame_id}.png", format='png')
