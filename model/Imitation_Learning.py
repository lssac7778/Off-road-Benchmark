import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
from tensorboardX import SummaryWriter
import common.const as const
from common.utills import print_square
from model.networks import ImpalaBlock
import torchvision.transforms as transforms
import copy
from PIL import Image
from carla_game.carla_gamev09 import CarlaEnv

def numpy_to_pil(image_):
    image = copy.deepcopy(image_)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image *= 255
    image = image.astype('uint8')

    im_obj = Image.fromarray(image)
    return im_obj


class ImpalaCNN(nn.Module):
    def __init__(self, steering_n=1, throttle_n=1):
        super(ImpalaCNN, self).__init__()

        feature_size = 13057

        self.conv_layer = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )

        '''
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=steering_n+throttle_n)
        )
        '''

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=512),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=steering_n+throttle_n)
        )

    def forward(self, img, scalar, get_feature=False):
        feature = self.conv_layer(img)
        total_feature = torch.cat((feature, scalar), 1)
        output = self.fc_layer(total_feature)
        if get_feature:
            return output, feature
        else:
            return output

class Base:
    def __init__(self,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True,
                 eval_num=20):

        self.network = ImpalaCNN()
        self.input_shape = input_shape
        self.num_action = num_action
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.train_test_split = train_test_split

        self.eval_num = eval_num

        self.optimizer = Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.loss_func = nn.L1Loss()

        if len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

        self.use_cuda = use_cuda

        if use_cuda:
            self.network.cuda()

    def save(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        device = torch.device('cpu')
        self.network.load_state_dict(torch.load(path, map_location=device))

    def train_step(self, measurement):
        raise NotImplementedError("Please Implement this method")

    def eval_step(self, measurement):
        raise NotImplementedError("Please Implement this method")

    def preprocess_measurement(self, measurement):
        if 'aug_imgs' in measurement.keys():
            aug_imgs_ = measurement['aug_imgs']
            temp_size = list(aug_imgs_.size())
            aug_imgs = []
            for j in range(temp_size[1]):
                temp = []
                for i in range(temp_size[0]):
                    temp.append(torch.unsqueeze(aug_imgs_[i][j], 0))
                temp = torch.cat(temp, 0)
                aug_imgs.append(temp)

            for i, imgs in enumerate(aug_imgs):
                measurement['aug_imgs{}'.format(i)] = imgs
            measurement['aug_len'] = len(aug_imgs)
        return measurement

    def train(self, dataset):
        self.total_step = 0
        dataset_len = len(dataset)
        for epoch in range(self.max_epoch):
            for step, measurement in enumerate(dataset):
                if step == dataset_len - 1:
                    break
                self.total_step += 1

                measurement = self.preprocess_measurement(measurement)

                '''train valid split'''

                origin_measurement, eval_measurement = {}, {}
                for key in measurement.keys():
                    if type(measurement[key]) == torch.Tensor or type(measurement[key]) == np.ndarray:
                        point = int(self.train_test_split * len(measurement[key]))
                        origin_measurement[key] = measurement[key][:point]
                        eval_measurement[key] = measurement[key][point:]
                    else:
                        origin_measurement[key] = measurement[key]
                        eval_measurement[key] = measurement[key]

                '''train'''
                verbose = self.train_step(origin_measurement)


                '''valid'''
                eval_verbose = self.eval_step(eval_measurement)
                verbose.update(eval_verbose)

                '''verbose'''
                for metric in verbose.keys():
                    self.writer.add_scalar(metric, verbose[metric], self.total_step)

                verbose["epoch"] = epoch
                verbose["step/epoch"] = "{}/{}".format(step, dataset_len)
                verbose["step"] = self.total_step
                print_square(verbose)

            '''save model'''
            self.save(self.save_dir + "epoch{}.pth".format(epoch + 1))

    def train_no_valid(self, dataset):
        self.total_step = 0
        dataset_len = len(dataset)
        for epoch in range(self.max_epoch):
            for step, measurement in enumerate(dataset):
                if step == dataset_len - 1:
                    break
                self.total_step += 1

                measurement = self.preprocess_measurement(measurement)
                '''train'''
                verbose = self.train_step(measurement)

                '''online evaluation'''
                if step % (dataset_len//self.eval_num) == 0:
                    online_eval_result = self.eval_online()
                    verbose.update(online_eval_result)

                '''verbose'''
                for metric in verbose.keys():
                    self.writer.add_scalar(metric, verbose[metric], self.total_step)

                verbose["epoch"] = epoch
                verbose["step/epoch"] = "{}/{}".format(step, dataset_len)
                verbose["total step"] = self.total_step
                print_square(verbose)

            '''save model'''
            self.save(self.save_dir + "epoch{}.pth".format(epoch+1))


    def eval_online(self):

        eval_max_episode = 3
        eval_max_step = 2000

        env = CarlaEnv(
            log_dir='./CarlaLog.txt',
            render=False,
            server_size=(4, 3),
            city_name='Offroad_1',
            weather='ClearNoon',
            plot=False
        )

        transform = transforms.Compose([
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=const.MEAN, std=const.STD)])

        mean_reward = []
        mean_step = []
        try:
            for episode in range(eval_max_episode):
                state = env.reset()
                episode_reward = 0
                step = 0
                done = False
                next_obs = None
                while not done:
                    step += 1
                    img, speed = state, env.epinfos["speed"]
                    img = np.transpose(img, (1, 2, 0))
                    img = numpy_to_pil(img)
                    img = transform(img).float().numpy()
                    img = np.expand_dims(img, 0)
                    img, speed = torch.Tensor(img), torch.Tensor([[speed]])

                    if self.use_cuda:
                        img = img.cuda()
                        speed = speed.cuda()

                    action = self.network(img, speed)
                    action = action.detach().cpu().numpy()
                    steering, throttle = action[0]
                    action = [throttle, steering]
                    action = np.array(action)

                    next_state, reward, done, _ = env.step(action)
                    if step > eval_max_step:
                        done = True

                    next_obs = next_state
                    episode_reward += reward
                    state = next_state
                mean_reward.append(reward)
                mean_step.append(step)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            env.close()

        mean_reward = sum(mean_reward)/len(mean_reward)
        mean_step = sum(mean_step)/len(mean_step)
        eval_result = {
            "online_eval/average_reward": mean_reward,
            "online_eval/average_step": mean_step
        }
        return eval_result

class Vanilla(Base):
    def __init__(self,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 discrete,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True):

        super(Vanilla, self).__init__(
            max_epoch,
            batch_size,
            train_test_split,
            save_dir,
            log_dir,
            lr,
            input_shape,
            num_action,
            use_cuda
        )

        self.discrete = discrete
        self.steering_values = torch.Tensor(const.DISCRETE_STEER)
        self.throttle_values = torch.Tensor(const.DISCRETE_THROTTLE)
        self.steering_n = const.DISCRETE_STEER_N
        self.throttle_n = const.DISCRETE_THROTTLE_N

        if self.use_cuda:
            self.steering_values = self.steering_values.cuda()
            self.throttle_values = self.throttle_values.cuda()

    def softmax_to_onehot(self, tensor):
        max_idx = torch.argmax(tensor, 1, keepdim=True)
        one_hot = torch.FloatTensor(tensor.shape)
        if self.use_cuda:
            one_hot = one_hot.cuda()
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        return one_hot

    def continuous_to_discrete(self, labels, ranges):
        '''
        change continuous values to onehot vector by ranges
        labels : torch Tensor
        ranges : torch Tensor
        '''
        ranges_len = len(ranges)
        labels_len = len(labels)
        #range_len x labels_len
        labels_mat = labels.reshape(1, labels_len).repeat(ranges_len, 1)
        ranges_mat = torch.transpose(ranges.reshape(1, ranges_len).repeat(labels_len, 1), 0, 1)

        indexs = torch.argmin(torch.abs(labels_mat - ranges_mat), dim=0)
        indexs = indexs.reshape(labels_len, 1)

        num_classes = ranges_len
        matrix = torch.arange(num_classes).reshape(1, num_classes)
        if self.use_cuda:
            matrix = matrix.cuda()
        one_hot_target = (indexs == matrix).float()
        return one_hot_target

    def discrete_to_continuous(self, softmax_output):
        data_len = softmax_output.size()[0]

        steering_prob = softmax_output[:, :self.steering_n]
        throttle_prob = softmax_output[:, self.steering_n:]
        steering_onehot = self.softmax_to_onehot(steering_prob)
        throttle_onehot = self.softmax_to_onehot(throttle_prob)

        steering_mat = self.steering_values.reshape(1, self.steering_n).repeat(data_len, 1)
        throttle_mat = self.throttle_values.reshape(1, self.throttle_n).repeat(data_len, 1)

        if self.use_cuda:
            steering_mat = steering_mat.cuda()
            throttle_mat = throttle_mat.cuda()

        steering = torch.sum(steering_mat * steering_onehot, 1)
        throttle = torch.sum(throttle_mat * throttle_onehot, 1)

        action = torch.cat((steering.reshape(data_len, 1), throttle.reshape(data_len, 1)), -1)
        return action

    def get_action(self, img_tensor, speed):
        if type(speed) != torch.Tensor:
            speed = torch.Tensor(speed)

        assert type(img_tensor) == torch.Tensor
        assert list(img_tensor.size())[-3:] == [3, 120, 160], "input shape mismatch : {}".format(list(img_tensor.size()))

        with torch.no_grad():
            actions = self.network.forward(img_tensor, speed)
            if self.discrete:
                actions = self.discrete_to_continuous(actions)
        return actions

    def eval_step(self, measurement):
        with torch.no_grad():

            img_tensor = measurement['img'].float()
            speed = measurement['speed'].float()
            target_action = measurement['target_action'].float()

            if self.use_cuda:
                target_action = target_action.cuda()
                img_tensor = img_tensor.cuda()
                speed = speed.cuda()

            if self.discrete:
                steering = target_action[:, 0]
                throttle = target_action[:, 1]
                steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
                throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
                target_action = torch.cat((steering_discrete, throttle_discrete), -1)


            origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
            total_loss = self.loss_func(target_action, origin_action).mean()


            actions = self.get_action(img_tensor, speed)
            steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'eval_loss/total': total_loss.item(),
            'eval_action/steering': np.mean(steering),
            'eval_action/throttle': np.mean(throttle)
        }
        return verbose

    def train_step(self, measurement):

        '''get inputs and labels'''

        img_tensor = measurement['img'].float()
        speed = measurement['speed'].float()
        target_action = measurement['target_action'].float()

        if self.use_cuda:
            target_action = target_action.cuda()
            img_tensor = img_tensor.cuda()
            speed = speed.cuda()

        if self.discrete:
            steering = target_action[:, 0]
            throttle = target_action[:, 1]
            steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
            throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
            target_action = torch.cat((steering_discrete, throttle_discrete), -1)

        '''vanilla loss'''

        origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
        total_loss = self.loss_func(target_action, origin_action).mean()

        '''optimize'''

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        '''verbose'''

        actions = self.get_action(img_tensor, speed)
        steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'loss/total': total_loss.item(),
            'action/steering': np.mean(steering),
            'action/throttle': np.mean(throttle)
        }
        del total_loss
        return verbose


class Augment(Vanilla):
    def __init__(self,
                 randomizer,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 discrete,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True):

        super(Augment, self).__init__(
            max_epoch,
            batch_size,
            train_test_split,
            save_dir,
            log_dir,
            lr,
            discrete,
            input_shape,
            num_action,
            use_cuda
        )

        self.randomizer_class = randomizer
        self.randomizer = self.randomizer_class()

    def train_step(self, measurement):
        if self.total_step%1000:
            self.randomizer = self.randomizer_class()

        '''get inputs and labels'''

        img_tensor = measurement['img'].float()
        speed = measurement['speed'].float()
        target_action = measurement['target_action'].float()

        batch_size = len(img_tensor)

        if self.use_cuda:
            target_action = target_action.cuda()
            img_tensor = img_tensor.cuda()
            speed = speed.cuda()

        if self.discrete:
            steering = target_action[:, 0]
            throttle = target_action[:, 1]
            steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
            throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
            target_action = torch.cat((steering_discrete, throttle_discrete), -1)

        '''image augmentation'''

        aug_imgs_ = []
        for i in range(measurement['aug_len']):
            temp = measurement['aug_imgs{}'.format(i)].float()
            aug_imgs_.append(temp)

        aug_imgs_ = torch.cat(aug_imgs_, 0)
        with torch.no_grad():
            aug_img_tensors = self.randomizer(aug_imgs_.cuda())

        aug_img_tensor = torch.cat(aug_img_tensors, 0)
        aug_len = int(len(aug_img_tensor)/batch_size)
        aug_speed = torch.cat([speed for _ in range(aug_len)], 0)
        aug_target_action = torch.cat([target_action for _ in range(aug_len)], 0)

        '''cuda & discrete'''

        if self.use_cuda:
            aug_img_tensor = aug_img_tensor.cuda()
            aug_speed = aug_speed.cuda()
            aug_target_action = aug_target_action.cuda()

        '''detach for memeory leak prevent'''

        img_tensor.detach()
        speed.detach()
        target_action.detach()

        aug_img_tensor.detach()
        aug_speed.detach()
        aug_target_action.detach()

        '''vanilla loss'''

        origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
        origin_loss = self.loss_func(target_action, origin_action).mean()

        '''augment loss'''

        aug_action, aug_feature = self.network(aug_img_tensor, aug_speed, get_feature=True)
        aug_loss = self.loss_func(aug_target_action, aug_action).mean()

        '''feature loss'''

        repeated_origin_feature = torch.cat([origin_feature for _ in range(aug_len)], 0)
        feature_loss = self.loss_func(repeated_origin_feature, aug_feature).mean()

        '''optimize'''

        total_loss = origin_loss + aug_loss + feature_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        #tensor_visualizer(img_tensor, aug_img_tensor)
        #assert False

        '''verbose'''

        actions = self.get_action(img_tensor, speed)
        steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'loss/total': total_loss.item(),
            'loss/origin': origin_loss.item(),
            'loss/augment': aug_loss.item(),
            'loss/feature': feature_loss.item(),
            'action/steering': np.mean(steering),
            'action/throttle': np.mean(throttle)
        }
        return verbose