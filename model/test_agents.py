
import math
import numpy as np
import random
from carla_game.waypoints.waypoints import get_vector_from_degree, linear_transform
import math

class mathDriver:
    def __init__(self, speed_limits = [19, 20], back_mode=False):
        self.reset()
        self.back_mode = back_mode
        self.speed_limits = speed_limits

    def reset(self):
        self.throttle = 0.5
        self.back_count = 0
        self.prev_speed = 0
        self.prev_steer = 0

    def get_action(self, state):
        speed, labels, waypoints_heading, current_heading, distance_from_center = state
        speed *= 4

        '''steering'''

        if distance_from_center < 3.5:
            steering = sum(labels[1:4])/len(labels[1:4])
        else:
            steering = sum(labels[1:4]) / len(labels[1:4])

        '''throttle'''
        brake = 0
        if speed > self.speed_limits[1]:
            self.throttle -= 0.001
            brake = abs(speed - self.speed_limits[1])*0.5
        elif speed < self.speed_limits[0]:
            #self.throttle += 0.005 + abs(self.speed_limits[0] - speed)*0.1
            self.throttle += 0.01
        else:
            self.throttle -= 0.001

        if abs(steering) > 0.1 and speed > 5:
            brake = abs(steering)*2

        '''back mode'''

        if speed < 0.05 and self.prev_speed > 0.05 and self.back_mode:
            self.back_count = 10

        if self.back_count > 0:
            self.back_count -= 1

            if self.back_count == 0:
                self.throttle = 1
            else:
                self.throttle = -5
                steering *= -1
                if abs(steering) < 0.2:
                    steering = random.uniform(1, 1.5)
                    if random.randint(0, 1):
                        steering *= -1

        self.prev_speed = speed
        self.prev_steer = steering
        return self.throttle, steering, brake

class randomDriver:
    def __init__(self):
        self.throttle = 0.5

    def get_action(self, state):
        return self.throttle, random.uniform(-1, 1)

class crashDriver:
    def __init__(self):
        self.crash_speed = random.randint(5, 20)
        self.crash_mode = False

    def get_action(self, state):
        speed, labels, waypoints_heading, current_heading, distance_from_center = state
        speed *= 4

        '''steering'''
        if speed > self.crash_speed:
            self.crash_mode = True

        if not self.crash_mode:
            steering = sum(labels[1:4])/len(labels[1:4])
        else:
            steering = random.uniform(-1, 1)

        throttle = self.crash_speed/10

        return throttle, steering
