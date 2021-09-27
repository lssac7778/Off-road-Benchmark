import argparse
import common.const as const
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_handler.dataset import AugmentCarlaDataset
import model.Imitation_Learning as IL
from model.augment import Image_Randomizer

from common.utills import print_square, train_val_dataset, get_foldername, dict_to_txt
from common.arguments import get_parser

#from model.networks import ImpalaNetwork, LightNetwork
import os
import torch

def main(args):
    '''directory setting'''

    if args.discrete:
        action_type = "discrete"
    else:
        action_type = "continuous"

    env_name = get_foldername(args.data_path)
    id_string = "IL_{}_{}".format(env_name, action_type)
    id_string += "_" + args.id_string
    args.save_dir += "randomized/" + id_string + "/"
    args.log_dir += "randomized/" + id_string + "/"

    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    '''Dataset & Augmentation'''

    origin_transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)])

    aug_functions = [
        transforms.RandomAffine(30),
        transforms.RandomGrayscale(p=1),
        transforms.RandomPerspective(),
        transforms.RandomRotation(30, expand=False),
        transforms.ColorJitter(brightness=(0.2, 2),
                               contrast=(0.3, 2),
                               saturation=(0.2, 2),
                               hue=(-0.3, 0.3))
    ]
    aug_transformers = []
    for aug_func in aug_functions:
        transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            aug_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD)])
        aug_transformers.append(transform)

    transformers = {
        "origin": origin_transform,
        "augment": aug_transformers
    }

    train_data = AugmentCarlaDataset(
        data_dir=args.data_path,
        train=True,
        transform=transformers
    )

    Dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )

    '''verbose'''

    verbose = {}
    for key in vars(args):
        verbose[key] = getattr(args, key)
    verbose["data_len"] = len(train_data)
    verbose["batch_num"] = len(Dataloader)

    print_square(verbose)
    #dict_to_txt(verbose, args.log_dir + "args.txt")

    '''main loop'''
    if args.discrete:
        steering_n = const.DISCRETE_STEER_N
        throttle_n = const.DISCRETE_THROTTLE_N
    else:
        steering_n = 1
        throttle_n = 1

    model = IL.Augment(
        #network = ImpalaNetwork(steering_n=steering_n, throttle_n=throttle_n),
        randomizer = Image_Randomizer,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        train_test_split=args.train_ratio,
        save_dir = args.save_dir,
        log_dir = args.log_dir,
        lr = args.learning_rate,
        discrete = args.discrete,
        use_cuda = True
    )

    for key in verbose.keys():
        model.writer.add_text(key, str(verbose[key]), 0)

    if args.no_valid:
        model.train_no_valid(Dataloader)
    else:
        model.train(Dataloader)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    main(args)
