import torch
import torch.nn as nn
import cv2
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance

import torchvision.transforms as transforms

class Image_Randomizer:
    def __init__(self):
        self.randomizers = [Netrand_Randomizer()]

    def __call__(self, x):
        result = []
        for randomizer in self.randomizers:
            result += randomizer(x)
        return result

class Netrand_Randomizer:

    input_type = torch.Tensor

    def __init__(self, network_num=5, use_cuda=True):
        self.network_num = network_num
        self.use_cuda = use_cuda

        self.networks = [self.build_network() for _ in range(network_num)]

    def __call__(self, x):
        '''input single batch'''
        outputs = [net(x).detach() for net in self.networks]
        return outputs

    def build_network(self):
        network = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1))
        network.eval()
        if self.use_cuda:
            network.cuda()
        return network

class CV_Randomizer:
    def __init__(self):
        self.transformers = [
            transforms.RandomAffine(30),
            transforms.RandomCrop((20, 20)),
            transforms.RandomGrayscale(p=1),
            transforms.RandomPerspective(),
            transforms.RandomRotation(45, expand=False),
            transforms.ColorJitter(brightness=(0.2, 2),
                                   contrast=(0.3, 2),
                                   saturation=(0.2, 2),
                                   hue=(-0.3, 0.3))
        ]

    def __call__(self, x):
        outputs = [trans(x) for trans in self.transformers]
        return outputs
