import torch
import os
from common.utills import print_square, dict_to_txt
import torchvision.transforms as transforms
import common.const as const
import numpy as np
import argparse
from carla_game.carla_gamev09 import CarlaEnv
from main_A3C import A2C
from common.arguments import get_parser
from model.Imitation_Learning import numpy_to_pil

def main(args):

    '''create agent (see models)'''

    agent = A2C()
    agent.load_state_dict(torch.load(args.load_dir))

    '''make envs'''

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    '''make environment'''
    env = CarlaEnv(
        log_dir='./CarlaLog.txt',
        render=True,
        plot=False,
        server_size=(4, 3),
        city_name=args.city_name,
        weather=args.weather,
        is_image_state=True
    )

    '''print settings'''
    verbose = {}
    for key in vars(args):
        verbose[key] = getattr(args, key)
    print_square(verbose)

    '''main'''
    for i in range(args.max_episode):
        with torch.no_grad():
            state = env.reset()
            done = False
            while not done:
                speed = env.epinfos["speed"]
                state = np.transpose(state, (1, 2, 0))
                state = numpy_to_pil(state)
                state = transform(state).float()
                state = torch.unsqueeze(state, 0)

                action = agent.sample(state, torch.Tensor([[speed]]), deterministic=True)
                action = action.cpu().numpy()
                action = action[0]

                #action[0] = 0.5

                state, reward, done, info = env.step(action)
                verbose = {
                    "reward": reward,
                    "throttle": round(action[0], 2),
                    "steering": round(action[1], 2),
                    "mileage": env.epinfos["mileage"],
                    "speed": speed
                }

                print_square(verbose)
                env.render()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', default='./trained_models/', type=str)
    parser.add_argument('--city-name', default='Offroad_1', type=str)
    parser.add_argument('--weather', default='ClearNoon', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max-episode', default=10, type=int)

    args = parser.parse_args()
    main(args)