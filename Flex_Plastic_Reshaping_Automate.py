import subprocess
import datetime
import time
import argparse
import multiprocessing
from Util.post_training_process import *

if __name__ == '__main__':
    num_sample_per_iter = 2000
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='Name of the run', default='experiment')

    parser.add_argument('--env', help='environment ID', default='FlexPlasticReshaping-v5')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch_size_per_process',
                        help='Number of samples collected for each process at each iteration',
                        default=int(num_sample_per_iter))
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=1000)

    args = parser.parse_args()

    env_name = args.env
    seed = args.seed

    run_name = args.name
    for s in range(1):
        seed = s * 13 + 7 * (s ** 2)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        # i = 0
        seed_root_dir = 'data/local/' + str(st) + '_' + run_name + '_seed_' + str(
            seed)

        data_saving_path = 'data/local/' + str(st) + '_' + run_name + '_seed_' + str(
            seed) + '/ppo_' + env_name
        train_policy = subprocess.call(
            'OMP_NUM_THREADS="1" python ./running_regimes/Granular_Sweep_Training.py'
            + ' --env ' + args.env
            + ' --seed ' + str(seed)
            + ' --data_saving_path ' + str(data_saving_path + '/policy')
            + ' --batch_size_per_process ' + str(args.batch_size_per_process)
            + ' --num_iterations ' + str(args.num_iterations)
            + ' --run_dir ' + str(seed_root_dir)
            , shell=True)
