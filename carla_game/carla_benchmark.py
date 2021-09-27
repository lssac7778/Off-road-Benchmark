import random
from carla_game.carla_gamev09 import CarlaEnv
from carla_game.waypoints.waypoints import *
from common.utills import print_square
import csv
import os
import sys
import glob
import time
try:
    # carla 0.9버젼 라이브러리가 있는 곳을 작성
    pwd = os.getcwd()
    sys.path.append(glob.glob('%s/carla_v09/dist/carla-*%d.%d-%s.egg' % (
        pwd,
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

    import carla
    from carla import ColorConverter as cc
except IndexError:
    raise RuntimeError('cannot find carla directory')

class CarlaBenchmark(CarlaEnv):
    def __init__(self,
                 log_dir,
                 data_dir=None,
                 host='localhost',
                 port=2000,
                 server_path=None,
                 server_size=(400, 300),
                 image_size=(800, 600),
                 fps=10,
                 city_name='Offroad_1',
                 weather='ClearNoon',
                 render=True,
                 plot=True,
                 is_image_state=True
                 ):

        super(CarlaBenchmark, self).__init__(
             log_dir,
             data_dir=data_dir,
             host=host,
             port=port,
             server_path=server_path,
             server_size=server_size,
             image_size=image_size,
             fps=fps,
             city_name=city_name,
             weather=weather,
             render=render,
             plot=plot,
             is_image_state=is_image_state
        )

    def get_reward_done(self, epinfos):
        done = False
        success = 0

        if epinfos["nospeedtime_step"] > 20 and epinfos["current_step"] > 40:
            done = True
            success = -1

        if epinfos["distance_from_center"] > epinfos["track_width"]/2:
            done = True
            success = -1

        if epinfos['is_collision']:
            done = True
            success = -1

        if self.target_mileage == -1:
            target_mileage = self.city_lenghts[self.city_name]
        else:
            target_mileage = self.target_mileage

        if epinfos["mileage"] > target_mileage:
            done = True
            success = 1

        return success, done



class Benchmark:
    def __init__(self,
                 agent,
                 log_dir,
                 result_dir,
                 city_names,
                 weathers,
                 lane_distances,
                 trial,
                 data_dir=None,
                 host='localhost',
                 port=2000,
                 server_path=None,
                 server_size=(400, 300),
                 image_size=(800, 600),
                 fps=10,
                 render=True,
                 plot=True,
                 is_image_state=True
                 ):

        self.env_inputs = [
            log_dir,
            data_dir,
            host,
            port,
            server_path,
            server_size,
            image_size,
            fps,
            '',
            '',
            render,
            plot,
            is_image_state
        ]

        self.city_names = city_names
        self.weathers = weathers
        self.lane_distances = lane_distances
        self.trial = trial
        self.env = None
        self.result_dir = result_dir
        self.agent = agent
        self.render = render

        self.reset_result()
        self.append_csv(self.result.keys())

    def reset_result(self):
        self.result = {
            "city_name": [],
            "weather": [],
            "lane_distance": [],
            "is_whole": [],
            "init_pos": [],
            "target_pos": [],
            "success": [],
            "epinfos": []
        }

    def append_csv(self, array):
        with open(self.result_dir, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(array)

    def append_result(self, city_name, weather, lane_distance, is_whole, init_pos, target_pos, success, epinfos):
        self.result["city_name"].append(city_name)
        self.result["weather"].append(weather)
        self.result["lane_distance"].append(lane_distance)
        self.result["is_whole"].append(is_whole)
        self.result["init_pos"].append(init_pos)
        self.result["target_pos"].append(target_pos)
        self.result["success"].append(success)
        self.result["epinfos"].append(epinfos)
        self.append_csv([self.result[key][-1] for key in self.result.keys()])
        self.reset_result()

    def remake_env(self, city_name, weather):
        del self.env
        self.env_inputs[8] = city_name
        self.env_inputs[9] = weather
        self.env = CarlaBenchmark(*self.env_inputs)

    def evaluate(self, city_name, weather, lane_distance):

        #set initial position of the car
        if lane_distance!=-1:
            is_whole = False
        else:
            is_whole = True

        #verbose
        verbose = {
            "city_name": city_name,
            "weather": weather,
            "lane_distance": lane_distance,
            "is_whole": is_whole
        }
        #print_square(verbose)

        #loop episode
        state = self.env.reset()
        self.env.target_mileage = lane_distance
        success, done, epinfos = 0, False, None
        while not done:
            action = self.agent(state, success, done, epinfos)
            state, success, done, epinfos = self.env.step(action)
            if self.render:
                self.env.render()

        if success==1:
            success = True
        elif success==-1:
            success = False
        else:
            raise ValueError("success value is wrong")
        #append result
        self.append_result(
            city_name,
            weather,
            lane_distance,
            is_whole,
            self.env.init_pos,
            None,
            success,
            self.env.epinfos
        )

        verbose["success"] = success
        return verbose

    def main(self):
        total_trial_num = len(self.city_names) * len(self.weathers) * len(self.lane_distances) * self.trial
        current_trial_num = 0
        for city_name in self.city_names:
            for weather in self.weathers:
                self.remake_env(city_name, weather)
                try:
                    for lane_distance in self.lane_distances:
                        for t in range(self.trial):
                            current_trial_num += 1

                            '''
                            if current_trial_num < 1679:
                                continue
                            '''

                            start_time = time.time()

                            while True:
                                try:
                                    verbose = self.evaluate(city_name, weather, lane_distance)
                                    break
                                except Exception as e:
                                    print("Error Occured")
                                    print(e)


                            verbose["progress"] = "{}/{}".format(current_trial_num, total_trial_num)
                            verbose["trial/second"] = time.time() - start_time
                            print_square(verbose)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt")
                finally:
                    self.env.close()






