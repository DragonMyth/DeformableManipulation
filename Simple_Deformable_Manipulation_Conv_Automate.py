import subprocess
import datetime
import time
import argparse
import multiprocessing
from Util.post_training_process import *

if __name__ == '__main__':
    cpu_count = 8  # multiprocessing.cpu_count()
    num_sample_per_iter = 20000
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SimplerParticleCarving-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch_size_per_process',
                        help='Number of samples collected for each process at each iteration',
                        default=int(num_sample_per_iter / cpu_count))
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=1000)
    args = parser.parse_args()

    env_name = args.env
    seed = args.seed

    for s in range(1):
        seed = s * 13 + 7 * (s ** 2)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        # i = 0
        curr_run = str(s)
        seed_root_dir = 'data/local/' + str(st) + '_' + env_name + '_seed_' + str(
            seed)

        data_saving_path = 'data/local/' + str(st) + '_' + env_name + '_seed_' + str(
            seed) + '/ppo_' + env_name + '_run_' + str(curr_run)
        train_policy = subprocess.call(
            'OMP_NUM_THREADS="1" mpirun -np ' + str(
                cpu_count) + ' python ./running_regimes/Simple_Deformable_Manipulation_Conv_Training.py'
            + ' --env ' + args.env
            + ' --seed ' + str(seed)
            + ' --curr_run ' + curr_run
            + ' --data_saving_path ' + str(data_saving_path + '/policy')
            + ' --batch_size_per_process ' + str(args.batch_size_per_process)
            + ' --num_iterations ' + str(args.num_iterations)
            + ' --run_dir ' + str(seed_root_dir)
            , shell=True)
