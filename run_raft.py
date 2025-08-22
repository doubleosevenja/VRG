import os
import sys
import torch
import numpy as np
import cv2

sys.path.append('core')  # 加入 RAFT 的 core 目录

from raft import RAFT
from core.utils.utils import InputPadder
from core.utils.flow_viz import flow_to_image
from argparse import Namespace

MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def preprocess(img):
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img


class RAFTWrapper:
    def __init__(self, model_path='raft-things.pth'):
        args = Namespace(small=False, mixed_precision=False, alternate_corr=False)
        self.model = RAFT(args)
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()

    def predict(self, pre_frame, cur_frame):
        image1 = preprocess(pre_frame).cuda()
        image2 = preprocess(cur_frame).cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

        return flow_up[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
