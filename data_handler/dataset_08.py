import os
import sys
import glob
import copy
import math
import random
import csv
import numpy as np

# from torch.utils.data import Dataset
import torch
import cv2
from tqdm import tqdm

# Semantic Segmentation classes
classes = {
    0: [0, 0, 0],         # None
    1: [70, 70, 70],      # Buildings
    2: [190, 153, 153],   # Fences
    3: [72, 0, 90],       # Other
    4: [220, 20, 60],     # Pedestrians
    5: [153, 153, 153],   # Poles
    6: [157, 234, 50],    # RoadLines
    7: [128, 64, 128],    # Roads
    8: [244, 35, 232],    # Sidewalks
    9: [107, 142, 35],    # Vegetation
    10: [0, 0, 255],      # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]     # TrafficSigns
}


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, train=True, model_name=None, split_size=0.88):
        self.data_dir = data_dir
        self.train = train
        self.img_size = (88, 200)
        self.split_size = split_size
        self.model_name = model_name

        if self.train:
            self.dir = os.path.join(self.data_dir, 'train')
        else:
            self.dir = os.path.join(self.data_dir, 'test')
        
        self.img_data, self.target_data = self._process(self.dir)   # type : list

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        # img --> RGB image, measurements --> [steer, throttle, speed]
        try:
            measurement = {
                'img': self.img_data[index],
                'speed': np.array([self.target_data[index][2]]),
                'direction': np.array([self.target_data[index][3]]),
                'target_action': self.target_data[index][:2]
            }
            
        except IndexError:
            print("Blank Image")
            measurement = {
                'img': np.zeros((17, self.img_size[0], self.img_size[1])),
                'speed': np.array([0.0]),
                'direction': np.array([-1.0]),
                'target_action': np.array([[0.0, 0.0]])
            }
        return measurement

    @staticmethod
    def get_measurements(path):
        measurements = []
        with open(glob.glob(os.path.join(path, 'measurements*'))[0], 'r') as ofd:
            w = csv.DictReader(ofd)

            for row in w:
                steer = float(row['steer'])
                if float(row['throttle']) >= 0.0:
                    throttle = float(row['throttle'])
                else:
                    throttle =  float(row['brake'])
                
                command = float(row['direction'])
                
                if 'speed' in row.keys():
                    speed = float(row['speed'])    
                else:
                    speed = float((row[None])[0])

                measurements.append(np.array([steer, throttle, speed, command]))

        return measurements

    def _process(self, dir_path):
        episode_lst = glob.glob(os.path.join(dir_path, 'episode_*'))
        if len(episode_lst) == 0:
            return None, None
        random.shuffle(episode_lst)
        
        imgs, measurements = [], []
        append = imgs.append
        for episode in tqdm(episode_lst):
            c_rgb_lst = glob.glob(os.path.join(episode, 'rgb_*'))
            c_semseg_lst = glob.glob(os.path.join(episode,'semseg_*'))
            c_depth_lst = glob.glob(os.path.join(episode, 'depth_*'))
            measurements = self.get_measurements(episode)

            for rgb, semseg, depth in zip(c_rgb_lst, c_semseg_lst, c_depth_lst):
                
                rgb = cv2.imread(rgb)
                rgb = rgb[:, :, ::-1]
                rgb = rgb.astype(np.float32)
                rgb /= 255.0
                
                semseg = cv2.imread(semseg)
                semseg = semseg[:, :, ::-1]
                semseg = self.cityscape_to_index(semseg)
                semseg = semseg.astype(np.float32)
                semseg /= 255.0

                depth = cv2.imread(depth)
                depth = depth[:, :, ::-1]
                depth = depth[:, :, 0]

                out = np.dstack((rgb, semseg, depth))
                out = np.transpose((2, 0, 1))
                append(out)
        return imgs, measurements
    
    @staticmethod
    def cityscape_to_index(image):
        result = np.zeros((image.shape[0], image.shape[1], 13))
        where = np.where
        items = classes.items()
        for key, value in items:
            x, y, z = where(image == value)
            result[x, y, key] = 1
        return result
