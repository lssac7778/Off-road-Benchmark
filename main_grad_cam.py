#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

import common.const as const
from PIL import Image
import argparse
import copy
import os.path as osp
import os

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
import gc

import model.Imitation_Learning as IL
from model.vanilla_ppo import Vanilla_PPO
from main_A3C import A2C
from main_LBC import Actor as LBC_actor

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

def get_gradcam(gcam, raw_image, paper_cmap=False):
    cmap = cm.jet_r(gcam)[..., :3] * 255.0 #120,160,3
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return gcam


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def numpy_to_pil(image_):
    image = copy.deepcopy(image_)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image *= 255
    image = image.astype('uint8')

    im_obj = Image.fromarray(image)
    return im_obj

def get_IL_model(load_dir):
    agent = IL.Vanilla(
        max_epoch=1,
        batch_size=1,
        save_dir="",
        log_dir="",
        lr=1,
        discrete=False,
        train_test_split=0.9,
        use_cuda=False
    )
    agent.load(load_dir)
    model = agent.network
    return model

def get_RL_model(load_dir):
    agent = Vanilla_PPO(is_image_state=True, lr=1, n_obs_per_round=1, batch_size=1)
    agent.load(load_dir)
    actor = torch.nn.Sequential(
        nn.Linear(in_features=13057, out_features=256, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=256, out_features=2, bias=True)
    )
    actor.load_state_dict(agent.network.actor.fc_layer.state_dict())
    class model_class(nn.Module):
        def __init__(self,):
            super(model_class, self).__init__()
            self.conv_layer = agent.network.perception.network
            self.fc_layer = actor

        def forward(self, img, scalar, get_feature=False):
            feature = self.conv_layer(img)
            total_feature = torch.cat((feature, scalar), 1)
            output = self.fc_layer(total_feature)
            if get_feature:
                return output, feature
            else:
                return output
    model = model_class()
    return model

def get_A3C_model(load_dir):
    agent = A2C()
    agent.load_state_dict(torch.load(load_dir))

    class model_class(nn.Module):
        def __init__(self):
            super(model_class, self).__init__()
            self.conv_layer = agent.perception
            self.fc_layer = agent.actor

        def forward(self, img, scalar, get_feature=False):
            feature = self.conv_layer(img)
            total_feature = torch.cat((feature, scalar), 1)
            output = self.fc_layer(total_feature)
            if get_feature:
                return output, feature
            else:
                return output
    model = model_class()
    return model

def get_LBC_model(load_dir):
    model = LBC_actor()
    model.load_state_dict(torch.load(load_dir))
    class model_class(nn.Module):
        def __init__(self):
            super(model_class, self).__init__()
            self.conv_layer = model.perception.network
            self.fc_layer = model.fc_layer

        def forward(self, img, scalar, get_feature=False):
            feature = self.conv_layer(img)
            total_feature = torch.cat((feature, scalar), 1)
            output = self.fc_layer(total_feature)
            if get_feature:
                return output, feature
            else:
                return output
    model = model_class()
    return model

def get_grad_cam(PIL_img, transform, model, target_layer, speed, gcam, cuda=False):
    raw_img_np = np.asarray(PIL_img, dtype=np.float32)
    img = transform(PIL_img).float().numpy()
    img = np.expand_dims(img, 0)

    inputs = [torch.Tensor(img), torch.Tensor([[speed]])]
    if cuda:
        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()


    # Images
    #images, raw_images = load_images(image_paths)
    #images = torch.stack(images).to(device)
    #
    # """
    # Common usage:
    # 1. Wrap your model with visualization classes defined in grad_cam.py
    # 2. Run forward() with images
    # 3. Run backward() with a list of specific classes
    # 4. Run generate() to export results
    # """

    # =========================================================================
    #bp = BackPropagation(model=model)
    #logits = bp.forward(*inputs)  # sorted

    # =========================================================================


    _ = gcam.forward(*inputs)

    #gbp = GuidedBackPropagation(model=model)
    #_ = gbp.forward(*inputs)

    # Guided Backpropagation
    #gbp.backward()
    #gradients = gbp.generate()

    # Grad-CAM
    #gcam.backward(ids=ids[:, [i]])
    gcam.backward()
    regions = gcam.generate(target_layer=target_layer)
    if cuda:
        regions = regions.cpu()

    # Grad-CAM
    regions = np.array(regions[0][0]*255, dtype=np.uint8)
    regions = cv2.resize(regions, dsize=tuple(reversed(raw_img_np.shape[:2])))
    regions = np.array(regions, dtype=np.float32)
    regions /= 255

    gcam_result = get_gradcam(
        gcam=regions,
        raw_image=raw_img_np,
    )
    return gcam_result



class Gcam_generator:
    def __init__(self, target_layer, model_type, cuda=True):

        if model_type=='IL':
            self.model = get_IL_model("./trained_models/vanilla/IL_Offroad1_continuous_online_eval/epoch1.pth")
            self.transform = transforms.Compose([
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const.MEAN, std=const.STD)
            ])
        elif model_type=='RIL':
            self.model = get_IL_model("./trained_models/randomized/IL_Offroad1_continuous_online_eval/epoch1.pth")
            self.transform = transforms.Compose([
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const.MEAN, std=const.STD)
            ])
        elif model_type=='RL':
            self.model = get_RL_model("./trained_models/vanilla/RL_Offroad_1__20210323-20h05m34s_Complete/model.pth")
            self.transform = transforms.Compose([
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const.MEAN, std=const.STD)
            ])
        elif model_type == 'A3C':
            self.model = get_A3C_model("./trained_models/A3C/impala_cnn_fixed/model_250000.pth")
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        elif model_type == 'LBC':
            self.model = get_LBC_model("./trained_models/vanilla/LBC_5_6/model.pth")
            self.transform = transforms.Compose([
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const.MEAN, std=const.STD)
            ])
        elif model_type == 'RLBC':
            self.model = get_LBC_model("./trained_models/randomized/LBC_5_6/model.pth")
            self.transform = transforms.Compose([
                transforms.Resize((const.HEIGHT, const.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=const.MEAN, std=const.STD)
            ])

        self.cuda = cuda

        if self.cuda:
            self.model.cuda()
        self.model.eval()

        self.target_layer = target_layer

        self.gradcam = GradCAM(model=self.model)

    def __call__(self, numpy_image, speed=3):

        PIL_image = Image.fromarray(numpy_image)
        gcam = get_grad_cam(PIL_image, self.transform, self.model, self.target_layer, speed, self.gradcam, self.cuda)
        return np.uint8(gcam)



def main(target_layer, input_dir, output_dir, cuda, gradcam, transform):

    device = get_device(cuda)

    model.to(device)
    model.eval()

    image_folder = input_dir
    image_paths = os.listdir(image_folder)
    image_paths.sort()

    speed = 3

    real_output_dir = output_dir + target_layer + "/"

    try:
        os.makedirs(real_output_dir)
    except:
        print("already results exists")
        print(output_dir, target_layer)
        print()
        return

    print("generating Grad-CAM")
    print(output_dir, target_layer)
    print()

    for k, image_name in enumerate(image_paths):
        image_path = image_folder + "/" + image_name

        raw_img = Image.open(image_path)

        gcam = get_grad_cam(raw_img, transform, model, target_layer, speed, gradcam, cuda)

        cv2.imwrite(
            osp.join(
                real_output_dir,
                "{}-gradcam.png".format(
                    image_name[:-4]#, classes[ids[j, i]]
                ),
            ),
            np.uint8(gcam)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target-layer", default='conv_layer.2.res2', type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="./figs/Grad-CAM/")
    parser.add_argument("--cuda", action='store_true')
    args = parser.parse_args()

    layers = [
        "conv_layer.0.res2",
        "conv_layer.1.conv1",
        "conv_layer.1.res2",
        "conv_layer.2.conv1",
        "conv_layer.2.res2"
    ]

    models = [
        get_IL_model("./trained_models/vanilla/IL_Offroad1_continuous_online_eval/epoch1.pth"),
        get_IL_model("./trained_models/randomized/IL_Offroad1_continuous_online_eval/epoch1.pth"),
        get_A3C_model("./trained_models/A3C/impala_cnn_fixed/model_250000.pth"),
        get_LBC_model("./trained_models/vanilla/LBC_5_6/model.pth"),
        get_LBC_model("./trained_models/randomized/LBC_5_6/model.pth")
    ]

    input_dir = "./figs/inputs"

    output_dirs = [
        "./figs/Grad-CAM/vanilla_IL_online_eval/",
        "./figs/Grad-CAM/randomized_IL_online_eval/",
        "./figs/Grad-CAM/vanilla_RL_A3C/",
        "./figs/Grad-CAM/vanilla_LBC_5_6/",
        "./figs/Grad-CAM/randomized_LBC_5_6/"
    ]

    vanilla_transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)
    ])

    gray_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    transforms = [
        vanilla_transform,
        vanilla_transform,
        gray_transform,
        vanilla_transform,
        vanilla_transform
    ]

    for model, output_dir, transform in zip(models, output_dirs, transforms):
        gradcam = GradCAM(model=model)
        for layer in layers:
            main(layer, input_dir, output_dir, args.cuda, gradcam, transform)
