import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from model.networks import LargeImpalaCNN, LargeImpalaBlock

from common.arguments import get_parser_rl
import numpy as np
import torchvision.transforms as transforms
import common.const as const
from carla_game.carla_gamev09 import CarlaEnv
from common.utills import print_square
from model.test_agents import mathDriver, randomDriver, crashDriver
from model.Imitation_Learning import numpy_to_pil
import argparse
import ray
import time
from tensorboardX import SummaryWriter
import copy
import gc
import os

def set_gradients(gradients, model):
    for param, grad in zip(model.parameters(), gradients):
        param._grad = grad

def get_gradients(model):
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.clone())
    return gradients

class A2C(nn.Module):
    def __init__(self, n_actions=2):
        super(A2C, self).__init__()

        '''
        feature_size = 11265
        self.perception = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        '''
        feature_size = 13057
        self.perception = nn.Sequential(
            LargeImpalaBlock(in_channels=1, out_channels=16),
            LargeImpalaBlock(in_channels=16, out_channels=32),
            LargeImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=n_actions),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )
        log_std = -0.5 * np.ones(n_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, state, speed):
        feature = self.perception(state)
        total_feature = torch.cat((feature, speed), 1)
        value = self.critic(total_feature)
        action_mean = self.actor(total_feature)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        return dist, value
    
    def sample(self, state, speed, deterministic=False):
        feature = self.perception(state)
        total_feature = torch.cat((feature, speed), 1)
        action_mean = self.actor(total_feature)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        if not deterministic:
            action = dist.sample()
        else:
            action = action_mean
        return action
    
    def evaluate(self, state, speed, actions):
        dist, value = self.forward(state, speed)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logprobs, entropy, value

@ray.remote(num_gpus=1)
class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""
    def __init__(self, env, actor_id):
        # starts simulation environment, policy, and thread.
        # Thread will continuously interact with the simulation environment
        #self.env = env
        #self.env_args = env
        self.env = CarlaEnv(**env)
        self.done = True
        self.id = actor_id
        self.network = A2C()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.last_state = None

    def get_rollouts(self, total_step):
        if not self.done:
            state = self.last_state

        try:
            buffer = []
            for _ in range(total_step):
                if self.done:
                    #self.env = CarlaEnv(**self.env_args)
                    state = self.env.reset()
                
                state = np.transpose(state, (1, 2, 0))
                state = numpy_to_pil(state)
                state = self.transform(state).float()
                state = torch.unsqueeze(state, 0)

                speed = self.env.epinfos["speed"]
                speed = torch.Tensor([[speed]])
                if self.is_cuda:
                    state = state.cuda()
                    speed = speed.cuda()
                action_tensor = self.network.sample(state, speed, deterministic=False)
                action = action_tensor.clone().cpu().detach().numpy()[0]

                next_state, reward, self.done, _ = self.env.step(action)
                buffer.append([state, speed, action, reward, self.done])
                state = next_state
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            self.env.close()
            
        self.last_state = next_state
        return buffer
    
    def compute_gradient(self, params, hyperparams):
        n_step = hyperparams["n_step"]
        gamma = hyperparams["gamma"]
        value_coef = hyperparams["value_coef"]
        entropy_coef = hyperparams["entropy_coef"]
        is_cuda = hyperparams["is_cuda"]
        self.is_cuda = is_cuda

        self.network.load_state_dict(params)
        self.network.eval()

        if self.is_cuda:
            self.network.cuda()

        buffer = self.get_rollouts(n_step)
        self.network.train()

        states, speeds, actions, rewards, dones = zip(*buffer)
        states, speeds, actions = (
            torch.stack(states),
            torch.stack(speeds),
            torch.tensor(actions)
        )

        states = torch.squeeze(states, dim=1)
        speeds = torch.squeeze(speeds, dim=1)

        action_logprobs, entropy, value = self.network.evaluate(states, speeds, actions)

        #compute returns
        returns = list(rewards)
        returns[-1] = value[-1].item()
        for i in range(n_step-2, -1, -1):
            returns[i] += (1-dones[i])*gamma*returns[i+1]
        returns = torch.tensor(returns)
        returns = torch.unsqueeze(returns, dim=-1)

        #compute loss
        advantages = returns - value
        value_loss = advantages.pow(2).mean()
        policy_loss = -(action_logprobs*advantages.detach()).mean()
        entropy = entropy.mean()
        total_loss = policy_loss + value_coef*value_loss - entropy_coef*entropy
        total_loss.backward()

        gradients = get_gradients(self.network)
        self.optimizer.zero_grad()
        
        info = {
            "id": self.id,
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "rewards": rewards,
            "dones": dones
        }

        del states, speeds, actions, params, returns, buffer
        return gradients, info
    
    def __del__(self):
        try:
            self.env.close()
        except:
            pass

def main(args):
    ray.init()

    max_step = args.max_step
    learning_rate = args.learning_rate
    is_cuda = args.is_cuda
    n_step = args.n_step
    gamma = args.gamma
    value_coef = args.value_coef
    entropy_coef = args.entropy_coef
    num_workers = args.num_workers

    writer = SummaryWriter(args.log_dir)
    for key in vars(args):
        writer.add_text(key, str(getattr(args, key)), 0)

    ports = [2010 + i*10 for i in range(num_workers)]

    envs = [{
        "log_dir": './CarlaLog.txt',
        "render": args.render,
        "plot": args.plot,\
        "port": port,
        "server_size": (4, 3),
        "city_name": args.city_name,
        "weather": args.weather,
        "is_image_state": True
    } for port in ports]

    hyperparams = {
        "n_step": n_step,
        "gamma": gamma, 
        "value_coef": value_coef, 
        "entropy_coef": entropy_coef,
        "is_cuda": is_cuda
    }

    main_network = A2C()
    if is_cuda:
        main_network.cuda()
    optimizer = torch.optim.RMSprop(main_network.parameters(), lr=learning_rate)

    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    '''start agents'''

    hyperparams_id = ray.put(hyperparams)
    agents = [Runner.remote(envs[i], i) for i in range(num_workers)]
    parameters = main_network.state_dict()
    gradient_list = []
    for agent in agents:
        gradient_list.append(agent.compute_gradient.remote(
            parameters,
            hyperparams_id
        ))

    '''main loop'''
    reinit_iteration = 50000//n_step

    total_step = 0
    step_second_list = []
    iteration = 0
    while total_step < max_step:
        start_time = time.time()

        done_id, gradient_list = ray.wait(gradient_list)
        gradient, info = ray.get(done_id)[0]

        optimizer.zero_grad()
        set_gradients(gradient, main_network)
        torch.nn.utils.clip_grad_norm_(main_network.parameters(), args.max_grad_norm)
        optimizer.step()

        parameters = main_network.state_dict()
        gradient_list.extend([agents[info["id"]].compute_gradient.remote(parameters, hyperparams_id)])

        step_second = (time.time()-start_time)/n_step
        step_second_list.append(step_second)
        step_second_avg = sum(step_second_list)/len(step_second_list)
        avg_reward = sum(info["rewards"])/len(info["rewards"])

        print_square({
            "total_loss": info["total_loss"],
            "policy_loss": info["policy_loss"],
            "value_loss": info["value_loss"],
            "entropy": info["entropy"],
            "rewards": avg_reward,
            "total_step": total_step,
            "step/second": round(step_second, 3),
            "remaining hour": round(step_second_avg*(max_step-total_step)/60/60, 3)
        })
        writer.add_scalar("loss/total", info["total_loss"], total_step)
        writer.add_scalar("loss/policy", info["policy_loss"], total_step)
        writer.add_scalar("loss/value", info["value_loss"], total_step)
        writer.add_scalar("loss/entropy", info["entropy"], total_step)
        writer.add_scalar("rewards", avg_reward, total_step)

        if iteration%reinit_iteration == 0 and iteration>0:
            '''reinit ray'''
            ray.shutdown()
            gc.collect()
            ray.init()

            hyperparams_id = ray.put(hyperparams)
            agents = [Runner.remote(envs[i], i) for i in range(num_workers)]
            parameters = main_network.state_dict()
            gradient_list = []
            for agent in agents:
                gradient_list.append(agent.compute_gradient.remote(
                    parameters,
                    hyperparams_id
                ))

            #save model
            torch.save(main_network.state_dict(), args.save_dir + f"model_{total_step}.pth")

        total_step += n_step
        iteration += 1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city-name', default='Offroad_1', type=str)
    parser.add_argument('--weather', default='ClearNoon', type=str)
    parser.add_argument('--log_dir', default='./logs/A3C/', type=str)
    parser.add_argument('--save_dir', default='./trained_models/A3C/', type=str)
    parser.add_argument('--max_step', default=1_500_000, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--is_cuda', default=False, type=bool)
    parser.add_argument('--n_step', default=5, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--value_coef', default=0.5, type=float)
    parser.add_argument('--entropy_coef', default=0.01, type=float)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--plot', default=False, type=bool)
    parser.add_argument('--cuda', default=False, type=bool)
    args = parser.parse_args()
    main(args)