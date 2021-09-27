import torchvision.transforms as transforms
from carla_game.carla_gamev09 import CarlaEnv
import torch
import common.const as const
from common.utills import average_dicts
from multiprocessing import Process, Pipe
import multiprocessing as mp


def _subproc_worker(pipe, env):
    try:
        while True:
            cmd, data = pipe.recv()
            print(cmd)
            if cmd == 'reset':
                pipe.send(env.reset())
            elif cmd == 'step':
                pipe.send(env.step(data))
            elif cmd == 'get_epinfos':
                pipe.send(env.get_epinfos(data))
            elif cmd == 'get_verbose':
                pipe.send(env.get_verbose())
            elif cmd == 'close':
                env.close()
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class MultiVecCarlaGym:
    def __init__(self,
                 process_num,
                 ports,
                 log_dir='./CarlaLog.txt',
                 host='localhost',
                 server_size=(400, 300),
                 image_size=(800, 600),
                 fps=10,
                 city_name='Offroad_1',
                 wheather='ClearNoon',
                 render=False,
                 plot=False,
                 is_image_state=True
                 ):

        self.process_num = process_num
        self.ports = ports

        self.envs = []
        self.pipes = []
        self.processes = []

        self.ctx = mp.get_context('spawn')

        for i in range(process_num):
            env = CarlaEnv(
                log_dir=log_dir,
                port=self.ports[i],
                server_size=server_size,
                city_name=city_name,
                host=host,
                image_size=image_size,
                fps=fps,
                wheather=wheather,
                render=render,
                plot=plot,
                is_image_state=is_image_state
            )
            parent_conn, child_conn = self.ctx.Pipe()
            p = self.ctx.Process(target=_subproc_worker, args=(child_conn, env))
            self.envs.append(env)
            self.pipes.append(parent_conn)
            self.processes.append(p)

        self.start_process()

    def start_process(self):
        for p in self.processes:
            p.start()

    def join_process(self):
        for p in self.processes:
            p.join()

    def send(self, cmds, datas):
        for i, pipe in enumerate(self.pipes):
            pipe.send((cmds[i], datas[i]))

    def recv(self):
        results = []
        for pipe in self.pipes:
            results.append(pipe.recv())
        return results

    def reset(self):
        cmds = ['reset' for _ in range(self.process_num)]
        datas = [None for _ in range(self.process_num)]
        self.send(cmds, datas)
        return self.recv()

    def step(self, actions):
        '''
        action = [throttle, steering, brake]
        '''

        self.send('step', actions)
        samples = self.recv()

        states, rewards, dones, epinfos = [], [], [], []
        for i, (next_state, reward, done, epinfo) in enumerate(samples):
            if done:
                self.envs[i].set_position()
                next_state = self.envs[i].reset()
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            epinfos.append(epinfo)

        new_epinfos = {}
        for key in epinfos[0].keys():
            new_epinfos[key] = []
        for info in epinfos:
            for key in info.keys():
                new_epinfos[key].append(info[key])
        return states, rewards, dones, new_epinfos

    def get_epinfos(self, key):
        cmds = ['get_epinfos' for _ in range(self.process_num)]
        datas = [key for _ in range(self.process_num)]
        self.send(cmds, datas)
        return self.recv()

    def get_verbose(self):
        cmds = ['get_verbose' for _ in range(self.process_num)]
        datas = [None for _ in range(self.process_num)]
        self.send(cmds, datas)
        return average_dicts(self.recv())

    def close(self):
        cmds = ['close' for _ in range(self.process_num)]
        datas = [None for _ in range(self.process_num)]
        self.send(cmds, datas)
        return self.join_process()


class DummyVecCarlaGym:
    def __init__(self,
                 process_num,
                 ports,
                 log_dir='./CarlaLog.txt',
                 host='localhost',
                 server_size=(400, 300),
                 image_size=(800, 600),
                 fps=10,
                 city_name='Offroad_1',
                 wheather='ClearNoon',
                 render=False,
                 plot=False,
                 is_image_state=True
                 ):

        self.process_num = process_num
        self.ports = ports
        self.envs = [CarlaEnv(log_dir=log_dir,
                              port=self.ports[i],
                              server_size=server_size,
                              city_name=city_name,
                              host=host,
                              image_size=image_size,
                              fps=fps,
                              wheather=wheather,
                              render=render,
                              plot=plot,
                              is_image_state=is_image_state) for i in range(process_num)]

    def reset(self):
        temp = []
        for i in range(self.process_num):
            state = self.envs[i].reset()
            temp.append(state)
        return temp

    def step(self, action):
        '''
        action = [throttle, steering]
        '''
        states, rewards, dones, epinfos = [], [], [], []
        for i in range(self.process_num):
            next_state, reward, done, epinfo = self.envs[i].step(action[i])
            if done:
                self.envs[i].set_position()
                next_state = self.envs[i].reset()

            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            epinfos.append(epinfo)

        new_epinfos = {}
        for key in epinfos[0].keys():
            new_epinfos[key] = []
        for info in epinfos:
            for key in info.keys():
                new_epinfos[key].append(info[key])
        return states, rewards, dones, new_epinfos

    def get_epinfos(self, key):
        result = []
        for i in range(self.process_num):
            result.append(self.envs[i].epinfos[key])
        return result

    def get_verbose(self):
        verboses = []
        for i in range(self.process_num):
            verbose = self.envs[i].get_verbose()
            verboses.append(verbose)
        return average_dicts(verboses)

    def close(self):
        for i in range(self.process_num):
            self.envs[i].close()


if __name__ == '__main__':
    import time
    transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)])

    env = VecCarlaGym(process_num=16, transform=transform, city_name="Offroad_1")
    start = time.time()
    state = env.reset()
    print(state[0].size())
    print(state[1].size())
    print(time.time() - start)

    for _ in range(10):
        start = time.time()
        actions = torch.Tensor([[0.5, 0] for _ in range(env.process_num)])
        env.step(actions)
        print(time.time() - start)
    env.close()



