import math
import argparse
import numpy as np
import common.const as const
import torchvision.transforms as transforms

from carla_game.carla_gamev09 import CarlaEnv
import torch

import model.Imitation_Learning as IL
from model.Imitation_Learning import numpy_to_pil
from common.utills import print_square

import cv2



def main(args):
    """
    CARLA 험지 전용 Wrapper 
        log_dir : 서버 debug 용 log file(필요)
        data_dir : 데이터 저장용 file (선택) (default: None)
        host : client host name (선택) (default: localhost)
        port : client port number (선택) (default: 2000)
        server_path : server`s absolute path (선택) (default: CARLA_ROOT)
        server_size : size of server image (선택) (default: 400,300)
        image_size : size of client(pygame) image (선택) (default: 800,600)
        city_name : name of city name  
                (choose Offroad_1, Offroad_2, Offroad_3, Offroad_4, Track) (선택) (default: Offroad_2)
        render : 서버와 클라이언트의 render 여부(True/False) (선택) (default: True)
    """

    '''make environment'''
    env = CarlaEnv(log_dir='./CarlaLog.txt', render=args.render, server_size=(4,3), city_name=args.city_name, plot=False)

    '''torch transform'''

    transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)])

    '''load model'''
    args.discrete = "discrete" in args.load_dir
    if args.discrete:
        steering_n = const.DISCRETE_STEER_N
        throttle_n = const.DISCRETE_THROTTLE_N
    else:
        steering_n = 1
        throttle_n = 1

    model = IL.Vanilla(
        max_epoch=1,
        batch_size=1,
        save_dir="",
        log_dir="",
        lr=1,
        discrete=args.discrete,
        train_test_split=0.9,
        use_cuda = False
    )

    model.load(args.load_dir)

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
            """
            reset 기능
            state -> type : list 추가 가능
            state[0] -> vehicle information(type: dict)
                key                  value
                - transform         : (x, y, z)
                - velocity          : (x, y, z)
                - angular_velocity  : (x, y, z)
                - collision         : int/float
            state[1] -> image(type: dict)
                key                    value_size
                - rgb               : (84, 84, 3)
                - depth             : (84, 84, 3)
                - visual(=rgb)      : (84, 84, 3) --> pygame visualize 용으로 사용.
            """
            state = env.reset()
            episode_reward = 0
            step = 0
            done = False
            next_obs = None
            while not done:
                step += 1

                if args.render:
                    """
                    render 기능
                        wrapper 생성 시 render를 False로 했다면 render코드 작성해도 동작 안함
                    """
                    if episode % 100 == 0:
                        env.render(save=False, step=step, model="None")
                    else:
                        env.render()

                #obs = dict_state(state)

                img, speed = state, env.epinfos["speed"]
                img = np.transpose(img, (1,2,0))

                #cv2.imwrite("image.png", (img*255).astype('uint8'))

                img = numpy_to_pil(img)

                img = transform(img).float().numpy()

                img = np.expand_dims(img, 0)

                action = model.get_action(torch.Tensor(img), torch.Tensor([[speed]]))

                action = action.detach().cpu().numpy()
                steering, throttle = action[0]

                action = [throttle, steering]
                action = np.array(action)

                """
                    step 기능
                모델의 출력 사이즈가 3인 경우
                action 출력 순서 : throttle, brake, steer
                모델의 출력 사이즈가 2인 경우
                action 출력 순서 : (throttle, brake), steer
                변경 시 wrapper의 step 함수에서 변경 가능
                """
                next_state, reward, done, _ = env.step(action)

                '''reward processing'''

                #next_obs  = dict_state(next_state)
                next_obs = next_state
                total_step += 1
                episode_reward += reward
                state = next_state

                print_square({"reward":reward, "throttle": throttle, "steering": steering, "img mean": np.mean(img), "img std": np.std(img)})

            total_reward += episode_reward
            print("#{}-th Episode, Total_Reward {}".format(episode, episode_reward))

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        env.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', default='./trained_models/', type=str)
    parser.add_argument('--city-name', default='Track1', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max-episode', default=3, type=int)
    
    args = parser.parse_args()
    args.render = True

    main(args)


