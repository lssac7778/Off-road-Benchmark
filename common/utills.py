from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import json
import cv2
import numpy as np
import random

def tensor_imwrite(img, name):
    temp = np.transpose(img.cpu().numpy(), (1, 2, 0))
    temp = (temp * 255).astype(np.uint8)
    cv2.imwrite(name, temp)

def numpy_imwrite(img, name):
    temp = np.transpose(img, (1, 2, 0))
    temp = (temp * 255).astype(np.uint8)
    cv2.imwrite(name, temp)

def tensor_visualizer(img_tensor, aug_img_tensor, idx=-1):
    if idx == -1:
        idx = random.randint(0, len(img_tensor)-1)

    img_tensor_len = len(img_tensor)
    aug_len = int(len(aug_img_tensor)/len(img_tensor))
    tensor_imwrite(img_tensor[idx], "origin.png")
    for i in range(aug_len):
        tensor_imwrite(aug_img_tensor[idx + img_tensor_len*i], "aug{}.png".format(i))


def average_dicts(array):
    result = {}
    for key in array[0].keys():
        result[key] = []

    for dictt in array:
        for key in dictt.keys():
            result[key].append(dictt[key])

    for key in result:
        result[key] = sum(result[key]) / len(result[key])
    return result

def print_square(dictionary):
    for key in dictionary.keys():
        if "float" in str(type(dictionary[key])):
            newval = round(float(dictionary[key]), 4)
            dictionary[key] = newval

    front_lens = []
    back_lens = []
    for key in dictionary.keys():
        front_lens.append(len(key))
        back_lens.append(len(str(dictionary[key])))
    front_len = max(front_lens)
    back_len = max(back_lens)

    strings = []
    for key in dictionary.keys():
        string = "| {0:<{2}} | {1:<{3}} |".format(key, dictionary[key], front_len, back_len)
        strings.append(string)

    max_len = max([len(i) for i in strings])
    print("-"*max_len)
    for string in strings:
        print(string)
    print("-" * max_len)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['valid'] = Subset(dataset, val_idx)
    return datasets

def get_foldername(_path):
    if _path[-1]=="/":
        path = _path[:-1]
    else:
        path = _path[:]

    last_idx = -1
    for i in range(len(path)):
        if path[i] == "/":
            last_idx = i
    return path[last_idx+1:]

def dict_to_txt(dicti, path):
    with open(path, 'w') as file:
        file.write(json.dumps(dicti))