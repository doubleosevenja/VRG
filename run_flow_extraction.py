import os
import cv2
import numpy as np
import sys

# 添加项目路径
sys.path.append('./raft_optical_flow_project')
from raft_wrapper import FlowServer

# 数据路径
video_dir = '../data/videos'
frame_dir = '../data/frames'
flow_dir = '../data/flows'


def main():
    # 检查输入目录
    if not os.path.exists(video_dir):
        print(f"Error: Video directory {video_dir} does not exist!")
        return

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    # 初始化RAFT模型
    try:
        server = FlowServer()
        print("RAFT model loaded successfully!")
    except Exception as e:
        print(f"Failed to load RAFT model: {e}")
        return

    for i, video_name in enumerate(video_files):
        name = os.path.splitext(video_name)[0]
        video_path = os.path.join(video_dir, video_name)
        print(f"Processing video {i + 1}/{len(video_files)}: {video_path}")

        save_frame_dir = os.path.join(frame_dir, name)
        save_flow_dir = os.path.join(flow_dir, name)
        os.makedirs(save_frame_dir, exist_ok=True)
        os.makedirs(save_flow_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        ret, frame_prev = cap.read()
        if not ret:
            print(f"Failed to read first frame from {video_path}")
            continue

        frame_prev = cv2.resize(frame_prev, (512, 512))
        index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (512, 512))

            try:
                # 调用RAFT光流预测
                flow = server.predict(frame_prev, frame)

                # 保存帧和光流
                frame_save_path = os.path.join(save_frame_dir, f"{index:06d}.jpg")
                flow_save_path = os.path.join(save_flow_dir, f"{index:06d}.npy")

                cv2.imwrite(frame_save_path, frame)
                np.save(flow_save_path, flow)

                print(f"[{i + 1}] Frame {index}: flow shape {flow.shape}, min {flow.min():.2f}, max {flow.max():.2f}")

            except Exception as e:
                print(f"Error processing frame {index}: {e}")
                break

            index += 1
            frame_prev = frame

        cap.release()
        print(f"Completed video {i + 1}: {index} frames processed")


if __name__ == '__main__':
    main()