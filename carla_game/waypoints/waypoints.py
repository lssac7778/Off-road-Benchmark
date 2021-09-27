import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
import random

plt.ion()

class Waypoints:
    file_mapping = {
        "offroad_1": 'Offroad_1.csv',
        "offroad_2": 'Offroad_2.csv',
        "offroad_3": 'Offroad_3.csv',
        "offroad_4": 'Offroad_4.csv',
        "offroad_5": 'Offroad_5.csv',
        "offroad_6": 'Offroad_6.csv',
        "offroad_7": 'Offroad_7.csv',
        "offroad_8": 'Offroad_8.csv'
    }

    def __init__(self, city_name):
        try:
            self.raw_waypoints = pd.read_csv("carla_game/waypoints/" + self.file_mapping[city_name.lower()])
        except:
            self.raw_waypoints = pd.read_csv(self.file_mapping[city_name.lower()])

        self.city_name = city_name
        self.city_num = int(self.city_name[-1])

        #process cm to m
        self.point_columns_labels = []
        for col in self.raw_waypoints.columns:
            if '_id' not in str(col):
                self.point_columns_labels.append(str(col))

        self.raw_waypoints[self.point_columns_labels] /= 100

        nparray = self.raw_waypoints[self.point_columns_labels].to_numpy()
        self.total_min = np.min(nparray)
        self.total_max = np.max(nparray)

        #nums
        self.points_num = len(self.raw_waypoints)

    def get_wp(self, idx, key='middle', d=2):
        if type(idx) == list or type(idx) == tuple:
            result = []
            for idd in idx:
                result.append(self.get_wp(idd))
            return result
        else:
            point = self.raw_waypoints.iloc[idx]

            data = []
            for xyz in ['.x', '.y', '.z']:
                data.append(point[key+xyz])
            data = data[:d]
            return data

    def get_init_pos(self):
        index = random.randint(0, self.points_num - 1)
        point = self.raw_waypoints.iloc[index]

        idxs = self.get_nearest_waypoints_idx(index)
        prev, next = idxs[random.randint(0, len(idxs) - 1)]
        yaw = get_degree(self.get_wp(prev[-1]), self.get_wp(next[0]))
        init_pos = (point["middle.x"], point["middle.y"], point["middle.z"], yaw)

        paths = self.path_from_idxs(init_pos[0:2], idxs)
        return init_pos, paths

    def get_mileage(self, passed_wps_idxs):
        result = 0
        for i in range(len(passed_wps_idxs)-1):
            result += get_dist_bet_point(self.get_wp(passed_wps_idxs[i]), self.get_wp(passed_wps_idxs[i+1]))
        return result

    def get_track_width(self, location_wp_index):
        return get_dist_bet_point(self.get_wp(location_wp_index, key='side1'), self.get_wp(location_wp_index, key='side2'))

    def get_nearest_waypoints_idx(self, location_wp_index, k=10):
        raise NotImplementedError

    def get_all_wps(self):
        result = []
        for i in range(self.points_num):
            result.append(self.get_wp(i))
            result.append(self.get_wp(i, key='side1'))
            result.append(self.get_wp(i, key='side2'))
        return result

    def get_current_wp_index(self, location):
        wps = self.raw_waypoints[["middle.x", "middle.y"]].values
        return find_nearest_waypoints(wps, location, 1)[0]

    def path_from_idxs(self, location, idxs):
        paths = []
        for prev, next in idxs:

            temp = {
                "prev_wps": np.asarray(self.get_wp(prev)),
                "next_wps": np.asarray(self.get_wp(next)),
                "prev_idxs": prev,
                "next_idxs": next,
            }
            temp["heading"] = get_degree(temp["prev_wps"][-1], temp["next_wps"][0])
            temp["distance_from_next_waypoints"] = [get_dist_bet_point(wp, location) for wp in temp["next_wps"]]
            temp["heading_slope"] = get_slope(temp["prev_wps"][-1], temp["next_wps"][0])
            temp["heading_bias"] = get_bias(temp["heading_slope"], temp["next_wps"][0])

            temp["distance_from_center"] = get_dist_from_line(location, temp["heading_slope"], temp["heading_bias"])
            paths.append(temp)
        return paths

    def get_paths(self, location, location_wp_index, prev_location_wp_index):
        idxs = self.get_prev_next_waypoints_idx(location_wp_index, prev_location_wp_index)
        return self.path_from_idxs(location, idxs)

    def get_prev_next_waypoints_idx(self, location_wp_index, prev_location_wp_index):
        paths = self.get_nearest_waypoints_idx(location_wp_index)
        if any([prev_location_wp_index in prev for prev, next in paths]):
            pass

        elif any([prev_location_wp_index in next for prev, next in paths]):
            # reverse paths
            for i in range(len(paths)):
                prev, next = paths[i]
                paths[i] = list(reversed(next)), list(reversed(prev))
        '''
        else:
            raise RuntimeError("Worng location_wp_index, prev_location_wp_index : {}, {}".format(location_wp_index, prev_location_wp_index))
        '''


        return paths

class Waypoints_lanekeeping(Waypoints):

    def get_nearest_waypoints_idx(self, location_wp_index, k=20):
        result = []
        for i in range(location_wp_index-k, location_wp_index+k+1):

            if i < 0:
                index = self.points_num + i
            else:
                index = i

            index = index % self.points_num
            result.append(index)

        return [[result[:k], result[k+1:]]]

class Waypoints_forked(Waypoints):

    def __init__(self, city_name):
        super(Waypoints_forked, self).__init__(city_name)

        self.groups_num = len(set(self.raw_waypoints["group_id"]))

        # gather indexs by path
        self.wp_idxs_by_path = []
        for gid in range(self.groups_num):
            temp = []
            for i in range(self.points_num):
                point = self.raw_waypoints.iloc[i]
                if point["group_id"] == gid:
                    temp.append(i)

            self.wp_idxs_by_path.append(temp)

    def get_nearest_waypoints_idx(self, location_wp_index):

        for path in self.wp_idxs_by_path:
            if location_wp_index in path:
                current_path = path
                break

        end_point = self.raw_waypoints.iloc[current_path[-1]]
        start_point = self.raw_waypoints.iloc[current_path[0]]

        front_paths = []
        end_paths = []

        #get available paths.
        for i in range(self.points_num):
            if end_point["inter_id"] == self.raw_waypoints.iloc[i]["inter_id"]\
                    and end_point["group_id"] != self.raw_waypoints.iloc[i]["group_id"]:

                for path in self.wp_idxs_by_path:
                    if i in path:
                        temp_path = path
                        if path[-1] == i:
                            temp_path.reverse()
                        elif path[0] == i:
                            pass
                        else:
                            print(current_path, path, i, end_point["inter_id"])
                            assert False, "invaild waypoints csv"

                        front_paths.append(temp_path)

            elif start_point["inter_id"] == self.raw_waypoints.iloc[i]["inter_id"]\
                    and start_point["group_id"] != self.raw_waypoints.iloc[i]["group_id"]:
                for path in self.wp_idxs_by_path:
                    if i in path:
                        temp_path = path
                        if path[0] == i:
                            temp_path.reverse()
                        elif path[-1] == i:
                            pass
                        else:
                            print(current_path, path, i, start_point["inter_id"])
                            assert False, "invaild waypoints csv"

                        end_paths.append(temp_path)

        #set points seq through heading
        current_idx = current_path.index(location_wp_index)

        total_paths = []
        for front_path in front_paths:
            for end_path in end_paths:
                temp = end_path + current_path + front_path
                current_loc_idx = len(end_path) + current_idx
                prev_points = temp[:current_loc_idx]
                next_points = temp[current_loc_idx + 1:]
                total_paths.append([prev_points, next_points])

        #remove overlap
        for i in range(len(total_paths)):
            total_paths[i] = list(total_paths[i])
            total_paths[i][0] = tuple(total_paths[i][0])
            total_paths[i][1] = tuple(total_paths[i][1])
            total_paths[i] = tuple(total_paths[i])
        total_paths = list(set(tuple(total_paths)))
        return total_paths

def get_waypoints_manager(city_name):
    if int(city_name[-1]) > 4:
        return Waypoints_forked(city_name)
    else:
        return Waypoints_lanekeeping(city_name)




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
        array[:, 1],
        array[:, 0],
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

def get_bias(slope, point):
    b = -slope*point[0] + point[1]
    return b

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


if __name__ == "__main__":
    waypoints_manager = Waypoints_forked('Offroad_6')
    waypoints_manager.get_init_pos()
    

