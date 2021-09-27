import numpy as np
import csv
import math
import pathlib
import matplotlib.pyplot as plt
import copy

plt.ion()

class Waypoints:
    file_mapping = {
        "Offroad_1": 'Offroad_1.csv',
        "Offroad_2": 'Offroad_2.csv',
        "Offroad_3": 'Offroad_3.csv',
        "Offroad_4": 'Offroad_4.csv',
        "Offroad_5": 'OffRoad_5_waypoint.csv',
        "Offroad_6": 'OffRoad_6_waypoint.csv',
        "Offroad_7": 'OffRoad_7_waypoint.csv',
        "Offroad_8": 'OffRoad_8_waypoint.csv'
    }

    def __init__(self, city_name):

        file = self.file_mapping[city_name]
        self.wps = self.load_waypoints_new(pathlib.Path(__file__).parent.absolute().as_posix() + '/{}'.format(file))
        self.track_length = self.get_track_length()
        self.wp_num = len(self.wps["middle"])
        self.total_min = min([float(np.min(self.wps[key])) for key in self.wps.keys()])
        self.total_max = max([float(np.max(self.wps[key])) for key in self.wps.keys()])

    def load_waypoints_new(self, path):
        txts = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for txt in reader:
                txts.append(txt)

        keys = [
            "middle.x",
            "middle.y",
            "middle.z",
            "side1.x",
            "side1.y",
            "side1.z",
            "side2.x",
            "side2.y",
            "side2.z"
        ]
        middle_x = txts[0].index("middle.x")
        middle_y = txts[0].index("middle.y")
        middle_z = txts[0].index("middle.z")
        side1_x = txts[0].index("side1.x")
        side1_y = txts[0].index("side1.y")
        side1_z = txts[0].index("side1.z")
        side2_x = txts[0].index("side2.x")
        side2_y = txts[0].index("side2.y")
        side2_z = txts[0].index("side2.z")

        middle = np.array([[
            i[middle_x],
            i[middle_y],
            i[middle_z]
        ] for i in txts[1:]], dtype=np.float32)

        side1 = np.array([[
            i[side1_x],
            i[side1_y],
            i[side1_z]
        ] for i in txts[1:]], dtype=np.float32)

        side2 = np.array([[
            i[side2_x],
            i[side2_y],
            i[side2_z]
        ] for i in txts[1:]], dtype=np.float32)

        result = {
            "middle": middle[:, 0:2] / 100,
            "side1": side1[:, 0:2] / 100,
            "side2": side2[:, 0:2] / 100,
            "middle_z": middle[:, 2] / 100,
            "side1_z": side1[:, 2] / 100,
            "side2_z": side2[:, 2] / 100
        }

        return result

    def get_all_wps(self):
        result = []
        for key in self.wps.keys():
            if 'z' not in key:
                result += self.wps[key].tolist()
        result = np.array(result)
        return result

    def get_progress(self, nearest_wp):
        idx = np.where(self.wps["middle"] == nearest_wp)[0][0]
        processed = idx / len(self.wps["middle"])
        return processed

    def sort_wps(self, init_pos):
        wps = copy.deepcopy(self.wps["middle"])

        way_vector = get_vector_from_degree(init_pos["yaw"])

        init_point = np.array([init_pos["pos"][0], init_pos["pos"][1]])
        idxs = find_nearest_waypoints(wps, init_point, 1)
        idx = idxs[0]
        init_wp = self.get_wp(idx)

        way_init_wp = init_wp + way_vector * 100
        big = self.get_wp(idx + 5)

        delta = get_dist_bet_point(way_init_wp, init_wp)
        delta_new = get_dist_bet_point(way_init_wp, big)

        reverse = False
        if delta_new > delta:
            reverse = True

        for key in self.wps.keys():
            list_wps = self.wps[key].tolist()
            list_wps = list_wps[idx:] + list_wps[:idx]
            if reverse:
                list_wps = list_wps[1:] + [list_wps[0]]
                list_wps.reverse()
            self.wps[key] = np.array(list_wps)

    def get_wp(self, idx_, key='middle'):
        wps = self.wps[key]
        idx = int(idx_)
        max_idx = len(wps) - 1
        if abs(idx) > max_idx:
            temp = abs(idx) % max_idx
            if idx < 0:
                idx = temp * -1
            else:
                idx = temp
        if 0 > idx:
            idx = max_idx - abs(idx) + 1
        return wps[idx]

    def get_idx(self, wp, key='middle'):
        result = None
        for i in range(self.wp_num):
            if np.array_equal(self.wps[key][i], wp):
                result = i
        return result

    def get_nearest_waypoints(self, goal_index, k=4, key='middle'):
        next_p = [self.get_wp(goal_index + i, key) for i in range(k)]
        prev_p = [self.get_wp(goal_index - j - 1, key) for j in range(k)]
        return next_p, prev_p

    def get_track_length(self):
        total_dist = 0
        for i in range(len(self.wps)):
            total_dist += get_dist_bet_point(self.get_wp(i), self.get_wp(i + 1))
        return total_dist


class Animator:
    def __init__(self, figsize=(10, 10), lims=(-400, 400)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(lims)
        # for legend, expand y max limit
        self.ax.set_ylim([lims[0], lims[1]+70])

        self.points_controller = {}
        self.linear_controller = {}

    def plot_points(self, dictt):
        '''
        dictt[key] = [array, dotsize]
        '''
        for key in dictt:
            if key in self.points_controller.keys():
                self.points_controller[key].set_data(dictt[key][0][:, 1], dictt[key][0][:, 0])
            else:
                self.points_controller[key] = plot_points(* [self.ax]+dictt[key]+[key])

    def plot_linears(self, dictt):
        '''
        dictt[key] = [slope, bias, minv, maxv]
        '''
        for key in dictt:
            if key in self.linear_controller.keys():
                x, y = get_dots_from_linear(*dictt[key])
                self.linear_controller[key].set_data(y, x)
            else:
                self.linear_controller[key] = plot_linear(* [self.ax]+dictt[key]+[key])

    def update(self):
        self.ax.legend(fontsize=10, loc='upper left')
        self.fig.canvas.draw()

    def __del__(self):
        plt.close(self.fig)

def plot_points(ax, array, dotsize, label):
    data_setter = ax.plot(
        array[:, 0],
        array[:, 1],
        marker='o',
        linestyle='',
        markersize=dotsize,
        label=label
    )
    return data_setter[0]

def get_dots_from_linear(slope, bias, minv, maxv):
    linear = lambda x: x * slope + bias
    width = maxv - minv
    x = np.linspace(minv, maxv, width)
    y = linear(x)
    return x, y

def plot_linear(ax, slope, bias, minv, maxv, label=''):
    x, y = get_dots_from_linear(slope, bias, minv, maxv)
    return ax.plot(x, y, label=label)[0]


def get_dist_bet_point(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

def get_dist_from_line(point, slope, b):
    x, y = point[0], point[1]
    ax, by, c = slope, -1, b
    return abs(ax*x + by*y + c)/(ax**2 + by**2)**(1/2)

def get_slope(point1, point2):
    return (point1[1] - point2[1])/(point1[0] - point2[0])

def get_vertical_slope(point1, point2):
    return -1/get_slope(point1, point2)

def get_linear(point, slope):
    b = -slope*point[0] + point[1]
    return slope, b

def sign(num):
    if num==0:
        return 0
    result = int(num/abs(num))
    assert result==1 or result==-1, "sign error | num:{}, result:{}".format(num, result)
    return result

def find_nearest_waypoints(waypoints, location, k):
    num_wps = len(waypoints)

    repeated_location = np.repeat(np.expand_dims(location, 0), num_wps, axis=0)
    mse = np.sum((repeated_location - waypoints)**2, axis = 1)

    idx = np.argpartition(mse, k)
    return idx[:k]

def load_waypoints(path):
    txts = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        for txt in reader:
            txts.append(txt)

    x_idx = txts[0].index('location.x')
    y_idx = txts[0].index('location.y')

    waypoints = np.array([[i[x_idx], i[y_idx]] for i in txts[1:]], dtype=np.float32)
    return waypoints

def get_vector_from_degree(degree):
    radian = degree / 180 * 3.14
    return np.array([math.cos(radian), math.sin(radian)])

def linear_transform(basis_vector, vector):
    transformer = np.zeros((2, 2))
    transformer[0][0] = basis_vector[0]
    transformer[0][1] = basis_vector[1]
    transformer[1][0] = -basis_vector[1]
    transformer[1][1] = basis_vector[0]
    transformer = np.linalg.inv(transformer)

    # linear transformation
    new_way_vector = np.matmul(vector, transformer)
    return new_way_vector

def get_degree(prev_point, next_point):
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)
    #track_direction = track_direction if track_direction >= 0 else 180 - track_direction
    return track_direction


if __name__=="__main__":
    city_name = 'Offroad_2'

    for i in ENV_INIT_VALUES[city_name]:
        teacher = Teacher(city_name, i)
        print(teacher.unit)
        print(teacher.get_nearest_waypoints([100, 100], 4))
    

