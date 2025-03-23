import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import cv2
from glob import glob
from flownet2.flowlib import visualize_flow

np.random.seed(1001)



class VrData(Dataset):
    def __init__(self, training=True, time_step=16, use_flow=False):
        self.use_flow = use_flow
        flow_dir = './data/flows'
        frame_dir = './data/frames'
        label_path = './data/videos.csv'

        test_classes = ['helicoptercrash', 'dunerovers', 'cartooncoaster', 'dotsouter']
        train_classes = [cls for cls in os.listdir(frame_dir) if not cls in test_classes]

        classes = train_classes if training else test_classes


        print('classes', classes)

        df = pd.read_csv(label_path)
        names = df['Title']
        scores = df['Sickness Rating']
        name2score = {}
        for name, score in zip(names, scores):
            name2score[name] = score

        self.video_clips = []
        self.labels = []

        for cls in classes:
            cls_dir = os.path.join(frame_dir, cls)
            frames_path = glob(os.path.join(cls_dir, '*'))
            frames_path.sort()

            for i in tqdm(np.arange(0, len(frames_path)-time_step, time_step if not training else time_step), desc=cls):
                self.video_clips.append(frames_path[i:i+time_step])
                self.labels.append(name2score[cls])

    def load_clips(self, video_clip):
        frames = []
        for frame_path in video_clip:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (256,256))/255.0
            frames.append(frame)
        frames = np.array(frames)

        return frames.transpose(3,0,1,2)

    def load_flows(self, video_clip):
        flows = []
        for frame_path in video_clip:
            flow_path = frame_path.replace('frames', 'flows').replace('jpg', 'npy')
            flow = np.load(flow_path)
            # flow = cv2.resize(flow, (256,256))

            flows.append(flow)
        flows = np.array(flows)

        return flows.transpose(3,0,1,2)


    def __len__(self):
        return len(self.video_clips)
    def __getitem__(self, item):
        frames = self.load_clips(self.video_clips[item])
        label = self.labels[item]
        if self.use_flow:
            flows = self.load_flows(self.video_clips[item])
            return torch.tensor(frames).float(), torch.tensor(flows).float(), torch.tensor(label).float()

        return torch.tensor(frames).float(), torch.tensor(label).float()


if __name__=='__main__':
    use_flow = True
    dataset = VrData(use_flow=use_flow, training=False)
    dataloader = DataLoader(dataset, num_workers=16, batch_size=4, shuffle=True)


    for data in dataloader:
        if use_flow:
            frames, flows, labels = data
            print(frames.shape, flows.shape, labels)

            plt.subplot(121)
            plt.imshow(frames[0, :, 0].numpy().transpose(1,2,0))
            plt.subplot(122)
            visualize_flow(flows[0, :, 0].numpy().transpose(1,2,0))
            # plt.pause(0)
        else:
            frames, labels = data
            print(frames.shape, labels)


























