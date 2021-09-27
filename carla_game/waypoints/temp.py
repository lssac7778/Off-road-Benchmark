import pandas as pd
import csv
import matplotlib.pyplot as plt

from waypoints import Animator, Waypoints
import time

wpm = Waypoints("Offroad_8")
am = Animator(lims=(-200, 200))

for i in range(1, len(wpm.wps['middle'])):
    dictt = {'points': [wpm.wps['middle'][0:i], 10]}
    am.plot_points(dictt)
    am.update()
    time.sleep(0.1)