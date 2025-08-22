import argparse
import os
import sys
import cv2
import numpy as np
import torch
from core.raft import RAFT
from core.utils.utils import InputPadder


class FlowModelRAFT:
    def __init__(self, model_path):
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # RAFT参数
        self.args = argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0,
            cpu=True,
        )

        # 加载模型
        self.model = RAFT(self.args)

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # 处理多GPU训练的权重
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith('module.') else k
                new_state_dict[key] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            print(f"Model loaded from {model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, image1, image2):
        # 输入验证
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same shape")

        # BGR -> RGB，转tensor
        img1 = torch.from_numpy(image1[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(image2[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        # 填充到合适尺寸
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        with torch.no_grad():
            _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

        return flow


class FlowServer:
    def __init__(self):
        # 权重文件路径
        pretrained_path = './raft_optical_flow_project/pretrained/raft-sintel.pth'
        if not os.path.exists(pretrained_path):
            # 尝试其他可能路径
            alt_paths = [
                './pretrained/raft-sintel.pth',
                '../pretrained/raft-sintel.pth',
                './raft-sintel.pth'
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    pretrained_path = path
                    break
            else:
                raise FileNotFoundError("RAFT weights not found. Please download raft-sintel.pth")

        self.model = FlowModelRAFT(pretrained_path)

    def predict(self, pre_frame, cur_frame):
        return self.model.predict(pre_frame, cur_frame)


def main():
    # 简单测试
    parser = argparse.ArgumentParser(description='RAFT CPU flow extraction demo')
    args = parser.parse_args()

    try:
        flowserver = FlowServer()
        print("FlowServer initialized successfully!")

        # 如果有测试图片，可以测试
        test_img_path1 = '../example/0img0.ppm'
        test_img_path2 = '../example/0img1.ppm'

        if os.path.exists(test_img_path1) and os.path.exists(test_img_path2):
            cur_frame = cv2.imread(test_img_path1)
            pre_frame = cv2.imread(test_img_path2)

            print("Input frames shape:", pre_frame.shape)
            flow = flowserver.predict(pre_frame, cur_frame)
            print("Flow shape:", flow.shape)
            print("Flow min/max:", flow.min(), flow.max())
        else:
            print("Test images not found, but FlowServer is ready!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()