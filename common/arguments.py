
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='H', default='./data', dest='data_path', type=str)
    parser.add_argument('--save-dir' ,default='./trained_models/', type=str)
    parser.add_argument('--log-dir', default='./logs/', type=str)
    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--no_valid', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch_size', metavar='B', default=120, dest='batch_size', type=int)
    parser.add_argument('--number_of_workers', metavar='N', default=12, dest='number_of_workers', type=int)
    parser.add_argument('--max_epoch', metavar='E', default=5, dest='max_epoch' ,type=int)
    parser.add_argument('-lr', '--learning_rate', metavar='L', default=0.0001, type=float)
    parser.add_argument('-split', '--train_ratio', default=0.9, type=float)
    parser.add_argument('--id-string', default='', help='add string to id string')
    return parser

def get_parser_rl():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--city-name',
        default='Offroad_1',
        help='environment to train on (default: Offroad_1)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--id-string',
        default='',
        help='add string to id string')
    parser.add_argument(
        '--gpu',
        default=0,
        type=int)

    parser.add_argument('--image-state', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')

    parser.add_argument(
        '--target-n-obs',
        type=int,
        default=10_000_000,
        help='target observation (default: 10_000_000, CARLA paper setting)')

    parser.add_argument(
        '--n-minibatches',
        type=int,
        default=1,
        help='number of minibatches (default: 1)')

    parser.add_argument(
        '--n-steps',
        type=int,
        default=128,
        help='n steps bootstrap (default: 128)')

    parser.add_argument(
        '--init-port',
        type=int,
        default=2010,
        help='carla client-server port. start at init port, connect by increasing port number with interval (default: 2010)')

    parser.add_argument(
        '--port-interval',
        type=int,
        default=10,
        help='ports = [(init_port + interval * i) for i in (0, env_num)] (default: 10)')

    return parser