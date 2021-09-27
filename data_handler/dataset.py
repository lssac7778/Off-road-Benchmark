import sys
sys.path.append("~/ADD/carlaOffroad")

import os
import csv
import numpy as np
import common.const as const

from torch.utils.data import Dataset
import torch
from PIL import Image


class AugmentCarlaDataset(Dataset):
    def __init__(self, data_dir=None, train=True, model_name=None, split_size=0.88, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.img_size = (const.HEIGHT, const.WIDTH)
        self.split_size = split_size
        self.model_name = model_name
        self.transform = transform
        self.dir = self.data_dir
        self.target_data = self._process(self.dir)  # type : list

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        # img --> RGB image, measurements --> [steer, throttle, speed]
        try:
            pil_sample = self._pil_loader(self.target_data[index]['img_name'])
            # depth_sample = self._pil_loader(self.depth_data[index])

            sample = self.transform["origin"](pil_sample)
            aug_imgs = []
            for trans in self.transform["augment"]:
                aug_imgs.append(torch.unsqueeze(trans(pil_sample), 0))
            aug_imgs = torch.cat(aug_imgs, 0)
            # depth_sample = self.transform(depth_sample)

            measurement = {
                'img': sample,
                # 'depth_img': depth_sample,
                'aug_imgs': aug_imgs,
                'speed': np.array([self.target_data[index]['speed']]),
                'target_action': self.target_data[index]['target_actions']
            }

        except IndexError:
            print("Blank Image")
            measurement = {
                'img': np.zeros((3, self.img_size[0], self.img_size[1])),
                # 'depth_img': np.zeros((3, self.img_size[0], self.img_size[1])),
                'speed': np.array([0.0]),
                'target_action': np.array([[0.0, 0.0]])
            }
        return measurement

    @staticmethod
    def _pil_loader(path):
        img = Image.open(path).convert('RGB')
        return img

    @staticmethod
    def get_measurements(path):
        # throttles, steers, brakes, directions, speeds = [], [], [], [], []
        measurements = []

        with open(os.path.join(path, 'measurements_balanced.csv'), 'r') as ofd:
            w = csv.DictReader(ofd)

            for i, row in enumerate(w):
                if i < 2:
                    continue
                img_name = row['step'] + '.png'
                steer = float(row['steer'])
                if float(row['throttle']) >= 0.0:
                    throttle = float(row['throttle'])
                else:
                    throttle = float(row['brake'])

                speed = float(row['speed'])
                measurement = {
                    'img_name': os.path.join(path, img_name),
                    'speed': speed,
                    'target_actions': np.array([steer, throttle]),
                }
                measurements.append(measurement)

        return measurements

    def _process(self, dir_path):
        # episode_lst = dir_path
        #
        # if len(episode_lst) == 0:
        #     return None, None
        # random.shuffle(episode_lst)

        # for episode in tqdm(episode_lst):
        # c_rgb_list = sorted(glob.glob(os.path.join(dir_path, 'rgb_*')))
        # depth_list = sorted(glob.glob(os.path.join(dir_path, 'depth_*')))
        measurements = self.get_measurements(dir_path)

        return measurements






class CarlaDataset(Dataset):
    def __init__(self, data_dir=None, train=True, model_name=None, split_size=0.88, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.img_size = (const.HEIGHT, const.WIDTH)
        self.split_size = split_size
        self.model_name = model_name
        self.transform = transform
        self.dir = self.data_dir
        self.target_data = self._process(self.dir)  # type : list

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        # img --> RGB image, measurements --> [steer, throttle, speed]
        try:
            sample = self._pil_loader(self.target_data[index]['img_name'])
            # depth_sample = self._pil_loader(self.depth_data[index])
            if self.transform is not None:
                sample = self.transform(sample)
                # depth_sample = self.transform(depth_sample)

            measurement = {
                'img': sample,
                # 'depth_img': depth_sample,
                'speed': np.array([self.target_data[index]['speed']]),
                'target_action': self.target_data[index]['target_actions']
            }

        except IndexError:
            print("Blank Image")
            measurement = {
                'img': np.zeros((3, self.img_size[0], self.img_size[1])),
                # 'depth_img': np.zeros((3, self.img_size[0], self.img_size[1])),
                'speed': np.array([0.0]),
                'target_action': np.array([[0.0, 0.0]])
            }
        return measurement

    @staticmethod
    def _pil_loader(path):
        img = Image.open(path).convert('RGB')
        return img

    @staticmethod
    def get_measurements(path):
        # throttles, steers, brakes, directions, speeds = [], [], [], [], []
        measurements = []

        with open(os.path.join(path, 'measurements_balanced.csv'), 'r') as ofd:
            w = csv.DictReader(ofd)

            for i, row in enumerate(w):
                if i < 2:
                    continue
                img_name = row['step'] + '.png'
                steer = float(row['steer'])
                if float(row['throttle']) >= 0.0:
                    throttle = float(row['throttle'])
                else:
                    throttle = float(row['brake'])

                speed = float(row['speed'])
                measurement = {
                    'img_name': os.path.join(path, img_name),
                    'speed': speed,
                    'target_actions': np.array([steer, throttle]),
                }
                measurements.append(measurement)

        return measurements

    def _process(self, dir_path):
        # episode_lst = dir_path
        #
        # if len(episode_lst) == 0:
        #     return None, None
        # random.shuffle(episode_lst)

        # for episode in tqdm(episode_lst):
        # c_rgb_list = sorted(glob.glob(os.path.join(dir_path, 'rgb_*')))
        # depth_list = sorted(glob.glob(os.path.join(dir_path, 'depth_*')))
        measurements = self.get_measurements(dir_path)

        return measurements


class ImageDataset(Dataset):
    def __init__(self, data_dir=None, split_size=0.88, transform=None):
        self.data_dir = data_dir
        self.img_size = (const.HEIGHT, const.WIDTH)
        self.split_size = split_size
        self.transform = transform
        
        self.dir = self.data_dir
        
        self.target_data = self._process(self.dir)

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        # img --> RGB image, measurements --> [steer, throttle, speed]
        try:
            sample = self._pil_loader(self.target_data[index]['img_name'])
            # depth_sample = self._pil_loader(self.depth_data[index])
            if self.transform is not None:
                sample = self.transform(sample)
                # depth_sample = self.transform(depth_sample)

            measurement = {
                'img': sample
            }
            
        except IndexError:
            print("Blank Image")
            measurement = {
                'img': np.zeros((3, self.img_size[0], self.img_size[1]))
            }
        return measurement

    @staticmethod
    def _pil_loader(path):
        img = Image.open(path).convert('RGB')
        return img

    @staticmethod
    def get_measurements(path):
        # throttles, steers, brakes, directions, speeds = [], [], [], [], []
        measurements = []
        file_extentions = [".png", ".jpg"]

        with os.listdir(path) as file_name:
            if any([file_name.endswith(i) for i in file_extentions]):
                measurement = {
                    'img_name': os.path.join(path, file_name),
                }
                measurements.append(measurement)

        return measurements

    def _process(self, dir_path):
        measurements = self.get_measurements(dir_path)

        return measurements
