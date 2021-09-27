import argparse
import common.const as const
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from data_handler.dataset import CarlaDataset
import model.Imitation_Learning as IL

from common.utills import print_square, train_val_dataset, get_foldername, dict_to_txt
from common.arguments import get_parser

from model.Imitation_Learning import ImpalaCNN as ImpalaNetwork
import os

def main(args):

    '''directory setting'''

    if args.discrete:
        action_type = "discrete"
    else:
        action_type = "continuous"

    env_name = get_foldername(args.data_path)
    id_string = "IL_{}_{}".format(env_name, action_type)
    id_string += "_" + args.id_string
    args.save_dir += "vanilla/" + id_string + "/"
    args.log_dir += "vanilla/" + id_string + "/"

    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    '''Dataset'''

    transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)])

    train_data = CarlaDataset(
        data_dir=args.data_path,
        train=True,
        transform=transform
    )

    Dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.number_of_workers
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

    model = IL.Vanilla(
        #network = ImpalaNetwork(steering_n=steering_n, throttle_n=throttle_n),
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        train_test_split=args.train_ratio,
        save_dir = args.save_dir,
        log_dir = args.log_dir,
        lr = args.learning_rate,
        discrete = args.discrete
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