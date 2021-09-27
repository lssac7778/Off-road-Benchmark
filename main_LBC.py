
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from model.networks import LargeImpalaCNN

from common.arguments import get_parser_rl
import numpy as np
import torchvision.transforms as transforms
import common.const as const
from carla_game.carla_gamev09 import CarlaEnv
from common.utills import print_square
from model.test_agents import mathDriver, randomDriver, crashDriver
from model.Imitation_Learning import numpy_to_pil
import argparse
from tensorboardX import SummaryWriter

class Actor(nn.Module):
    def __init__(self, n_actions=2):
        super(Actor, self).__init__()
        feature_size = 13057
        self.perception = LargeImpalaCNN()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=n_actions)
        )
        '''
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=n_actions)
        )
        '''
        log_std = -0.5 * np.ones(n_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, state, speed, deterministic=True):
        feature = self.perception(state)
        total_feature = torch.cat((feature, speed), 1)
        action_mean = self.fc_layer(total_feature)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        if not deterministic:
            action = dist.sample()
        else:
            action = action_mean

        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action, action_logprobs.exp(), entropy

def main(args):
    '''make environment'''
    env = CarlaEnv(
        log_dir='./CarlaLog.txt',
        render=args.render,
        plot=args.plot,
        server_size=(4, 3),
        city_name=args.city_name,
        weather=args.weather,
        is_image_state=True
    )

    if args.randomization:
        transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.RandomChoice([
                transforms.RandomAffine(30),
                transforms.RandomPerspective(),
                transforms.RandomRotation(15, expand=False),
                transforms.ColorJitter(brightness=(0.2, 2),
                                       contrast=(0.3, 2),
                                       saturation=(0.2, 2),
                                       hue=(-0.3, 0.3)),
                transforms.Resize((const.HEIGHT, const.WIDTH))
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD),
        ])

    writer = SummaryWriter(args.log_dir)
    for key in vars(args):
        writer.add_text(key, str(getattr(args, key)), 0)

    #agent = crashDriver()
    teacher = mathDriver(speed_limits = [5, 6])
    actor = Actor()
    if args.test:
        actor.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.SGD(actor.parameters(), lr=args.learning_rate)
    loss_function = nn.SmoothL1Loss()

    if args.cuda:
        actor.cuda()

    '''verbose'''

    verbose = {}
    for key in vars(args):
        verbose[key] = getattr(args, key)
    print_square(verbose)

    '''main loop'''

    total_step = 0
    total_reward = 0
    try:
        for episode in range(args.max_episode):
            state = env.reset()
            episode_reward = 0
            step = 0
            done = False
            teacher.reset()
            while not done:
                step += 1

                if args.render:
                    if episode % 100 == 0:
                        env.render(save=False, step=step, model="None")
                    else:
                        env.render()

                state = np.transpose(state, (1, 2, 0))
                state = numpy_to_pil(state)
                state = transform(state).float()
                state = torch.unsqueeze(state, 0)

                _, _, sensors = env.get_full_observation()
                speed = sensors[0]
                speed = torch.Tensor([[speed]])
                if args.cuda:
                    state = state.cuda()
                    speed = speed.cuda()
                action_tensor, prob, entropy = actor(state, speed)
                action = action_tensor.clone().cpu().detach().numpy()[0]

                label_action_ = teacher.get_action(sensors)
                label_action_ = label_action_[0:2]
                label_action = torch.Tensor(label_action_)
                if args.cuda:
                    label_action = label_action.cuda()
                loss = loss_function(action_tensor, label_action)

                if not args.test:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                throttle, steering = action.tolist()

                brake = 0
                if abs(steering) > 0.1 and speed > 5:
                    brake += abs(steering) * 2
                if speed > 15:
                    brake += (speed-15)*4

                next_state, reward, done, _ = env.step([throttle, steering, brake])

                total_step += 1
                episode_reward += reward
                state = next_state

                '''
                print_square({
                    "reward": reward,
                    "label steering": label_action_[1],
                    "label throttle": label_action_[0],
                    "steering": action[1],
                    "throttle": action[0],
                    "loss": loss.item(),
                    "speed": env.epinfos["speed"],
                    "track_width": env.epinfos["track_width"],
                    "mileage": env.epinfos["mileage"],
                    "distance_from_center": env.epinfos["distance_from_center"],
                    "current_step": env.epinfos["current_step"],
                    "is_collision": env.epinfos["is_collision"],
                    "average_speed": env.epinfos["average_speed"]
                })
                '''

                writer.add_scalar("loss/total", loss.item(), total_step)
                writer.add_scalar("rewards", reward, total_step)

                if env.epinfos["current_step"] > 2000:
                    break

            total_reward += episode_reward
            writer.add_scalar("episode_rewards", episode_reward, total_step)
            print("#{}-th Episode, Total_Reward {}".format(episode, episode_reward))
            if not args.test:
                torch.save(actor.state_dict(), args.save_path)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        env.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city-name', default='Offroad_1', type=str)
    parser.add_argument('--weather', default='ClearNoon', type=str)
    parser.add_argument('--max-episode', default=50, type=int)
    parser.add_argument('--log_dir', default='./logs/vanilla/LBC/', type=str)
    parser.add_argument('--save_path', default='./logs/vanilla/LBC/model.pth', type=str)
    parser.add_argument('--load_path', default='./logs/vanilla/', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--randomization', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    args.render = False
    args.plot = False
    args.cuda = True
    main(args)