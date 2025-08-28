import sys

sys.path.append("./policy/CoA/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json
from scipy.spatial.transform import Rotation
from rlbench.demo import Demo
from rlbench.backend.observation import Observation

# 在文件开头添加导入
# 在文件开头添加导入
from src.envs.rlbench.rlbench_env import keypoint_discovery, _is_stopped

def check_keyframes_gripper_state(demo, episode_idx, output_file=None):
    """检查关键帧对应的gripper状态和停止状态"""
    # 调用keypoint_discovery获取关键帧索引
    keypoints = keypoint_discovery(demo, stopping_delta=0.01, method="heuristic")
    
    # # 准备输出内容
    output_lines = []
    output_lines.append(f"\n=== Episode {episode_idx} 关键帧分析 ===")
    output_lines.append(f"总帧数: {len(demo)}")
    output_lines.append(f"关键帧数量: {len(keypoints)}")
    output_lines.append(f"关键帧索引: {keypoints}")
    
    # print(f"\n=== Episode {episode_idx} 关键帧分析 ===")
    # print(f"总帧数: {len(demo)}")
    # print(f"关键帧数量: {len(keypoints)}")
    # print(f"关键帧索引: {keypoints}")
    
    # 检查每一帧的停止状态（不仅仅是关键帧）
    stopped_buffer = 0
    output_lines.append(f"\n=== 所有帧的停止状态分析 ===")
    
    for i in range(len(demo)):
        obs = demo[i]
        
        # 调用_is_stopped检查当前帧是否停止
        is_stopped_flag = _is_stopped(demo, i, obs, stopped_buffer, delta=0.1)
        stopped_buffer = 4 if is_stopped_flag else stopped_buffer - 1
        
        # 如果当前帧是关键帧或者is_stopped为True，则输出详细信息
        if i in keypoints or is_stopped_flag:
            frame_type = "关键帧" if i in keypoints else "普通帧"
            output_lines.append(f"\n{frame_type} (索引 {i}):")
            output_lines.append(f"  gripper_open: {obs.gripper_open}")
            output_lines.append(f"  gripper_pose: {obs.gripper_pose[:3]}")
            output_lines.append(f"  joint_velocities: {obs.joint_velocities}")
            
            if obs.gripper_open:
                output_lines.append("  → 左夹爪状态: 打开")
            else:
                output_lines.append("  → 左夹爪状态: 闭合")

    
    # 只关注关键帧的详细分析
    output_lines.append(f"\n=== 关键帧详细分析 ===")
    stopped_buffer = 0  # 重置buffer
    
    for i, keypoint_idx in enumerate(keypoints):
        if keypoint_idx < len(demo):
            obs = demo[keypoint_idx]
            
            # 检查关键帧的停止状态
            is_stopped_flag = _is_stopped(demo, keypoint_idx, obs, stopped_buffer, delta=0.1)
            stopped_buffer = 4 if is_stopped_flag else stopped_buffer - 1
            
            output_lines.append(f"\n关键帧 {i+1} (索引 {keypoint_idx}):")
            output_lines.append(f"  gripper_open: {obs.gripper_open}")
            output_lines.append(f"  gripper_pose: {obs.gripper_pose[:3]}")
            output_lines.append(f"  joint_velocities: {obs.joint_velocities}")
            
            if obs.gripper_open:
                output_lines.append("  → 左夹爪状态: 打开")
            else:
                output_lines.append("  → 左夹爪状态: 闭合")
            

    
    # 打印到控制台
    for line in output_lines:
        print(line)
    
    # 写入文件
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
            f.write('\n' + '='*50 + '\n')  # 分隔线
    
    return keypoints

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_endpose, left_gripper = (
            root["/endpose/left_endpose"][()],
            root["/endpose/left_gripper"][()],
        )


        right_endpose, right_gripper = (
            root["/endpose/right_endpose"][()],
            root["/endpose/right_gripper"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_endpose, left_gripper, right_endpose, right_gripper, image_dict




def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create variation0 directory structure
    variation_dir = os.path.join(save_path, "variation0")
    episodes_dir = os.path.join(variation_dir, "episodes")
    if not os.path.exists(episodes_dir):
        os.makedirs(episodes_dir)

    # Generate variation_descriptions.pkl
    # This file contains task variation descriptions for RLBench
    variation_descriptions = ["Default task variation"]
    variation_descriptions_path = os.path.join(variation_dir, "variation_descriptions.pkl")
    with open(variation_descriptions_path, "wb") as f:
        pickle.dump(variation_descriptions, f)
    print(f"Generated variation_descriptions.pkl at {variation_descriptions_path}")

    # 创建分析输出文件
    analysis_output = os.path.join(save_path, "keyframes_analysis.txt")

    # 清空文件
    if os.path.exists(analysis_output):
        open(analysis_output, 'w').close()

    for i in range(episode_num):
        left_endpose_all, left_gripper_all, right_endpose_all, right_gripper_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        
        # Create list to hold observations for this episode
        observations = []
        gripper_pose = []
        for j in range(0,left_gripper_all.shape[0]):
            # Decode images
            if j != left_gripper_all.shape[0] - 1:
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (128, 128))

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (128, 128))

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (128, 128))

                camera_front_bits = image_dict["front_camera"][j]
                camera_front = cv2.imdecode(np.frombuffer(camera_front_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_front_resized = cv2.resize(camera_front, (128, 128))

                # 处理双臂末端执行器数据
                left_endpose = left_endpose_all[j].astype(np.float32)   # 左臂末端位姿 (7维: x,y,z,qx,qy,qz,qw)
                # gripper_pose 包含了双臂的所有末端执行器数据
                gripper_pose = np.zeros(7, dtype=np.float32)
                gripper_pose[:7] = left_endpose           # 左臂末端位姿 7维 (x,y,z,qx,qy,qz,qw)
                gripper_open = True if left_gripper_all[j] > 0.5 else False  
                # 计算关节速度（通过时间差分）
                if j > 0:
                    prev_left_endpose = left_endpose_all[j - 1].astype(np.float32)

                    # 时间步长
                    dt = 1/250  # 仿真默认250Hz，对应0.004秒

                    joint_velocities = np.zeros(8, dtype=np.float32)

                    # 左臂线性速度
                    joint_velocities[:3] = (left_endpose[:3] - prev_left_endpose[:3]) / dt

                    # 左臂角速度
                    left_rot_prev = Rotation.from_quat(prev_left_endpose[3:7])
                    left_rot_curr = Rotation.from_quat(left_endpose[3:7])
                    left_rot_diff = left_rot_curr * left_rot_prev.inv()
                    joint_velocities[3:6] = left_rot_diff.as_rotvec() / dt
                    # gripper开合速度
                    joint_velocities[6] = (left_gripper_all[j] - left_gripper_all[j-1]) / dt
                    joint_velocities[7] = 0.0  # 预留或其他gripper相关速度
                else:
                    joint_velocities = np.zeros(8, dtype=np.float32)
                
                # gripper_matrix 作为 动作预测
                # 创建8x8矩阵来容纳双臂数据
                gripper_matrix = np.zeros((4, 4), dtype=np.float32)
                # 左臂齐次变换矩阵 (4x4)
                left_transform = np.eye(4, dtype=np.float32)
                left_transform[:3, :3] = Rotation.from_quat(left_endpose[3:7]).as_matrix()  # 旋转
                left_transform[:3, 3] = left_endpose[:3]  # 位置
                gripper_matrix[:4, :4] = left_transform
                gripper_matrix[3, 3] = left_gripper_all[j]  # 覆盖(3,3)位置存储夹爪状态

            # 以下变量未在训练中使用
            gripper_joint_positions = np.zeros(2, dtype=np.float32)
            joint_positions = np.zeros(8, dtype=np.float32)  # 初始化关节位置
            gripper_touch_forces = np.zeros(6, dtype=np.float32)
            task_low_dim_state = np.zeros(70, dtype=np.float32)
            joint_forces = np.zeros(16, dtype=np.float32)  # 力设为0
            # 准备深度图和掩码（如果没有实际数据，使用零数组）
            depth_shape = (128, 128)
            mask_shape = (128, 128)
            point_cloud_shape = (128*128, 3)  # 点云数据

            # 创建Observation对象，提供所有必需参数
            obs = Observation(
                left_shoulder_rgb=camera_left_wrist_resized,  # left_camera -> left_shoulder_rgb
                left_shoulder_depth=np.zeros(depth_shape, dtype=np.float32),
                left_shoulder_mask=np.zeros(mask_shape, dtype=np.uint8),
                left_shoulder_point_cloud=np.zeros(point_cloud_shape, dtype=np.float32),
                right_shoulder_rgb=camera_right_wrist_resized,  # right_camera -> right_shoulder_rgb
                right_shoulder_depth=np.zeros(depth_shape, dtype=np.float32),
                right_shoulder_mask=np.zeros(mask_shape, dtype=np.uint8),
                right_shoulder_point_cloud=np.zeros(point_cloud_shape, dtype=np.float32),
                overhead_rgb=camera_high_resized,  # head_camera -> overhead_rgb
                overhead_depth=np.zeros(depth_shape, dtype=np.float32),
                overhead_mask=np.zeros(mask_shape, dtype=np.uint8),
                overhead_point_cloud=np.zeros(point_cloud_shape, dtype=np.float32),
                wrist_rgb=camera_left_wrist_resized,  # 未使用,不包含在配置中
                wrist_depth=np.zeros(depth_shape, dtype=np.float32),
                wrist_mask=np.zeros(mask_shape, dtype=np.uint8),
                wrist_point_cloud=np.zeros(point_cloud_shape, dtype=np.float32),
                front_rgb=camera_front_resized,  # front_camera -> front_rgb
                front_depth=np.zeros(depth_shape, dtype=np.float32),
                front_mask=np.zeros(mask_shape, dtype=np.uint8),
                front_point_cloud=np.zeros(point_cloud_shape, dtype=np.float32),

                joint_velocities=joint_velocities,
                joint_positions=joint_positions,
                joint_forces=joint_forces,

                gripper_open=gripper_open,
                gripper_pose=gripper_pose,
                gripper_matrix=gripper_matrix,
                gripper_joint_positions=gripper_joint_positions,
                gripper_touch_forces=gripper_touch_forces,

                task_low_dim_state=task_low_dim_state,
                misc={}
            )
            
            observations.append(obs)

        # Create Demo object
        demo = Demo(observations)
        
        # Save demo in the correct directory structure
        episode_dir = os.path.join(episodes_dir, f"episode{i}")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)

        pickle_path = os.path.join(episode_dir, "low_dim_obs.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(demo, f)

        keypoints = check_keyframes_gripper_state(demo, i, output_file=analysis_output)
        
        begin += 1
        print(f"Processed and saved episode {i} to {pickle_path}")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    
    # The root directory for the raw HDF5 files
    # raw_data_dir = os.path.join("../../data/", task_name, task_config, 'singledata')
    raw_data_dir = "/home/lumina/lumina/wenjun/RoboTwin/data/move_can_pot/demo_clean/data" # TODO
    
    # The directory where the processed .pkl files will be saved
    # This path should correspond to the `dataset_root` in your launch.yaml
    # Need to include task_name subdirectory for RLBench compatibility
    save_dir = f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}/{task_name}"
    
    begin = 0
    begin=data_transform(
        raw_data_dir,
        expert_data_num,
        save_dir,
    )


    print("\nProcessing complete.")
    print(f"Processed data saved to: {os.path.abspath(save_dir)}")