# A New Open-Source Off-road Environment for Benchmark Generalization of Autonomous Driving

<img src = "https://user-images.githubusercontent.com/31644153/134851472-477c60e0-f1f7-4c16-8faf-efb1197ede1d.png" width="80%" height="80%">

[Isaac Han](https://github.com/lssac7778), [Dong-Hyeok Park](https://github.com/bhappy10), and [Kyung-Joong Kim](https://cilab.gist.ac.kr/hp/current-member/)

**IEEE Access** [\[Paper\]](https://ieeexplore.ieee.org/document/9552860) [\[Video\]](https://www.youtube.com/watch?v=SERSv0TFUwQ)

## Installation

1. download Off-road CARLA environment from [google drive](https://drive.google.com/file/d/1VqWp9lU5ysT1Pf9Z8Gm_y0rikp2vkgXO/view?usp=sharing)

2. clone this repository
```
git clone https://github.com/lssac7778/Off-road-Benchmark.git
```
3. pull docker image
```
docker pull lssac7778/carla
```

## Quick start
run CARLA server
```
sh <download-path>/CARLA_Shipping_0.9.6-dirty/LinuxNoEditor/CarlaUE4.sh -opengl
```
run test_env.py
```
cd Off-road-Benchmark
docker run -v $PWD:/app -e DISPLAY=$DISPLAY --net host --ipc host lssac7778/carla python test_env.py
```
## Document
1. [Train custom agent](https://github.com/lssac7778/Off-road-Benchmark/wiki#1-train-custom-agent)
2. [Evaluate custom agent](https://github.com/lssac7778/Off-road-Benchmark/wiki#2-evaluate-custom-agent)
3. [Use custom reward function](https://github.com/lssac7778/Off-road-Benchmark/wiki#3-use-custom-reward-function)
4. [How to import off-road maps into recent CARLA versions?](https://github.com/lssac7778/Off-road-Benchmark/blob/main/docs/Add_Custom_Map_to_Carla.pdf)


## Contact
If you have any questions about the paper or the codebase, please feel free to contact lssac7778@gm.gist.ac.kr
