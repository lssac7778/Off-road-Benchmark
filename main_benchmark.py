

import argparse

import torch

from carla_game.carla_benchmark import Benchmark
from model.test_agents import mathDriver
import model.Imitation_Learning as IL
import numpy as np
import torchvision.transforms as transforms
import common.const as const
from model.Imitation_Learning import numpy_to_pil
from model.vanilla_ppo import Vanilla_PPO
from main_LBC import Actor
from main_A3C import A2C

class AgentWarpper_mathDriver:
    def __init__(self):
        self.agent = mathDriver()

    def __call__(self, state, success, done, epinfos):
        return self.agent.get_action(state)

class AgentWarpper_IL:
    def __init__(self, load_dir):
        self.agent = IL.Vanilla(
            max_epoch=1,
            batch_size=1,
            save_dir="",
            log_dir="",
            lr=1,
            discrete=False,
            train_test_split=0.9,
            use_cuda=False
        )

        self.agent.load(load_dir)

        self.transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)])

    def __call__(self, state, success, done, epinfos):
        speed = 0
        if epinfos!=None:
            speed = epinfos["speed"]
        state = np.transpose(state, (1, 2, 0))
        state = numpy_to_pil(state)
        state = self.transform(state).float().unsqueeze(0)
        actions = self.agent.get_action(state, np.array([[speed]]))
        action = [float(actions[0][1]), float(actions[0][0])]
        return action

class AgentWarpper_RL:
    def __init__(self, load_dir):
        self.agent = Vanilla_PPO(is_image_state=True, lr=1, n_obs_per_round=1, batch_size=1)

        self.agent.load(load_dir)

        self.transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD)])

    def __call__(self, state, success, done, epinfos):
        speed = 0
        if epinfos != None:
            speed = epinfos["speed"]

        state = np.transpose(state, (1, 2, 0))
        state = numpy_to_pil(state)
        state = self.transform(state).float().unsqueeze(0)

        action, _, _, _ = self.agent.get_action([state, np.array([[speed]])])
        action = [float(action[0][0]), float(action[0][1])]
        return action

class AgentWarpper_A3C:
    def __init__(self, load_dir):
        self.agent = A2C()
        self.agent.load_state_dict(torch.load(load_dir))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __call__(self, state, success, done, epinfos):
        speed = 0
        if epinfos != None:
            speed = epinfos["speed"]

        state = np.transpose(state, (1, 2, 0))
        state = numpy_to_pil(state)
        state = self.transform(state).float().unsqueeze(0)

        action = self.agent.sample(state, torch.Tensor([[speed]]), deterministic=True)
        action = action.detach().cpu().numpy()
        action = [float(action[0][0]), float(action[0][1])]
        return action


class AgentWarpper_LBC:
    def __init__(self, load_dir):
        self.agent = Actor()
        self.agent.load_state_dict(torch.load(load_dir))

        self.transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD)])

    def __call__(self, state, success, done, epinfos):
        speed = 0
        if epinfos != None:
            speed = epinfos["speed"]

        state = np.transpose(state, (1, 2, 0))
        state = numpy_to_pil(state)
        state = self.transform(state).float().unsqueeze(0)

        action_tensor, _, _ = self.agent(state, torch.Tensor([[speed]]), deterministic=True)
        action = action_tensor.clone().cpu().detach().numpy().tolist()[0]
        return action


def main(args):

    city_names = [
        "Offroad_1",
        "Offroad_2",
        "Offroad_3",
        "Offroad_4",
        "Offroad_5",
        "Offroad_6",
        "Offroad_7",
        "Offroad_8"
    ]

    weathers = [
        "ClearNoon",
        "WetCloudyNoon",
        "MidRainyNoon",
        "ClearSunset",
        "WetCloudySunset",
        "MidRainSunset"
    ]

    lane_distances = [
        100,
        200,
        300,
        400,
        500,
        600,
        -1
    ]

    trial = 10

    '''make environment'''
    if "IL_" in args.load_dir:
        print("load IL")
        agent = AgentWarpper_IL(args.load_dir)

    elif "RL_" in args.load_dir:
        print("load RL")
        agent = AgentWarpper_RL(args.load_dir)

    elif "LBC" in args.load_dir:
        print("load LBC")
        agent = AgentWarpper_LBC(args.load_dir)

    elif "A3C" in args.load_dir:
        print("load A3C")
        agent = AgentWarpper_A3C(args.load_dir)

    else:
        print(f"RuntimeError : invaild load dir : {args.load_dir}")
        raise RuntimeError

    env = Benchmark(
        agent=agent,
        log_dir='./CarlaLog.txt',
        result_dir=args.save_dir,
        city_names=city_names,
        weathers=weathers,
        lane_distances=lane_distances,
        port=args.port,
        trial=trial,
        render=args.render,
        plot=args.plot,
        server_size=(4, 3),
        is_image_state=not args.sensor_state
    )

    '''main loop'''
    env.main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', default='./trained_models/', type=str)
    parser.add_argument('--save-dir', default='./tmp_benchmark.csv', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--sensor_state', action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    args = parser.parse_args()
    main(args)


