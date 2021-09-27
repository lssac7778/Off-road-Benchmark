import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as categorical
from torch.distributions import MultivariateNormal, Normal
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    def __init__(self):
        super(ImpalaCNN, self).__init__()
        self.network = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )
        self.output_size = 13056

    def forward(self, x):
        return self.network(x)


class LargeImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class LargeImpalaCNN(nn.Module):
    def __init__(self):
        super(LargeImpalaCNN, self).__init__()
        self.network = nn.Sequential(
            LargeImpalaBlock(in_channels=3, out_channels=16),
            LargeImpalaBlock(in_channels=16, out_channels=32),
            LargeImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )
    def forward(self, x):
        return self.network(x)


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Flatten()
        )
    def forward(self, x):
        return self.network(x)

class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
    def forward(self, x):
        return self.network(x)


class DiscreteActor(nn.Module):
    def __init__(self, n_actions, feature_size):
        super(DiscreteActor, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=n_actions)
        )
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, feature, deterministic=False):
        vector = self.fc(feature)
        action_probs = self.softmax(self.actor(vector))
        p = action_probs.exp()
        if deterministic:
            m, ix = torch.max(p, dim=-1)
        else:
            ix = categorical(p).sample()
        entropy = -(action_probs.exp() * (action_probs + 1e-8))
        entropy = entropy.sum(dim=1)
        entropy = entropy.mean()
        return ix, action_probs, entropy

class ContinuousActor(nn.Module):
    def __init__(self, n_actions, feature_size):
        super(ContinuousActor, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=n_actions),
        )

        log_std = -0.5 * np.ones(n_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, feature, deterministic):
        action_mean = self.fc_layer(feature)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        if not deterministic:
            action = dist.sample()
        else:
            action = action_mean

        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action, action_logprobs.exp(), entropy

    def evaluate(self, feature, action):
        action_mean = self.fc_layer(feature)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_logprobs.exp(), entropy

class MultiContinuousActor(nn.Module):
    def __init__(self, n_actions, feature_size, use_cuda=True):
        super(MultiContinuousActor, self).__init__()

        self.use_cuda = use_cuda

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=n_actions),
        )

        action_std = 0.5
        self.action_var = nn.Parameter(torch.full((n_actions,), action_std * action_std))
        #self.action_var = torch.full((n_actions,), action_std * action_std)

        #if self.use_cuda:
        #    self.action_var = self.action_var.cuda()

    def forward(self, feature, deterministic=False):
        action_mean = self.fc_layer(feature)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        if not deterministic:
            action = dist.sample()
        else:
            action = action_mean

        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action, action_logprobs.exp(), entropy

    def evaluate(self, feature, action):
        action_mean = self.fc_layer(feature)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        if self.use_cuda:
            cov_mat = cov_mat.cuda()

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs.exp(), dist_entropy