#!/usr/bin/env python3
import gym
import joblib
from gym import wrappers
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
import tensorflow as tf
import multiprocessing
from keras.models import load_model

import scipy.stats


def callback(localv, globalv):
    iters_so_far = localv['iters_so_far']
    local_kwargs = localv['kwargs']

    sess = local_kwargs['session']
    gap = local_kwargs['gap']
    save_dir = logger.get_dir()
    if (MPI.COMM_WORLD.Get_rank() == 0):
        if iters_so_far % gap == 0:
            save_dict = {}
            variables = localv['pi'].get_variables()
            for i in range(len(variables)):
                cur_val = variables[i].eval()
                save_dict[variables[i].name] = cur_val

            joblib.dump(save_dict, save_dir + '/policy_params_' + str(iters_so_far) + '.pkl', compress=True)


def train(sess, env_id, num_timesteps, timesteps_per_actor, seed, policy_param):
    from baselines.ppo1 import cnn_policy_carving, pposgd_simple, cnn_policy_carving_two_maps, \
        cnn_policy_carving_explicit_target

    rank = MPI.COMM_WORLD.Get_rank()

    workerseed = seed * 50 + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    env = gym.make(env_id)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        # return cnn_policy_carving.CnnPolicyCarving(name=name, ob_space=ob_space, ac_space=ac_space)
        return cnn_policy_carving_explicit_target.CnnPolicyCarvingExplicitTarget(name=name, ob_space=ob_space,
                                                                                 ac_space=ac_space)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))

    env.seed(seed + rank)

    model = pposgd_simple.learn(env, policy_fn,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=timesteps_per_actor,
                                clip_param=0.2, entcoeff=0,
                                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear',
                                callback=callback,
                                # Following params are kwargs
                                session=sess,
                                gap=5,
                                restore=None,  # [sess, policy_param]
                                )

    env.close()
    return model


def main(env_name, seed, run_num, data_saving_path, batch_size_per_process, num_iterations):
    num_processes = MPI.COMM_WORLD.Get_size()
    num_timesteps_per_process = batch_size_per_process
    num_iterations_enforce = num_iterations

    import baselines.common.tf_util as U

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    tf.reset_default_graph()

    with U.single_threaded_session() as sess:
        U.ALREADY_INITIALIZED.update(set(tf.global_variables()))
        logger.reset()
        logger.configure(data_saving_path)

        # filename = '/Users/dragonmyth/Documents/CS/DeformableManipulation/data/local/conv_policy_match_center_circle/ppo_SimplerParticleCarving-v2_run_0/policy/policy_params_500.pkl'
        # policy_param = joblib.load(filename)

        model = train(sess, env_name,
                      num_timesteps=num_iterations_enforce * num_processes * num_timesteps_per_process,
                      timesteps_per_actor=num_timesteps_per_process,
                      seed=seed, policy_param=None)

        if mpi_rank == 0:
            env = gym.make(env_name)
            if hasattr(env.env, 'disableViewer'):
                env.env.disableViewer = False
            env = wrappers.Monitor(env, logger.get_dir() + '/../results', force=True)

            obs = env.reset()

            step = 0
            while (True):

                env.render()
                actions = model._act(False, obs)
                obs, _, done, _ = env.step(actions[0][0])
                env.render()
                if done:
                    obs = env.reset()
                if done:
                    print("Visualization is Done")
                    break

                step += 1


if __name__ == '__main__':
    # main()
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', default='SimplerParticleCarving-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--curr_run', help='Current number of runs in the sequence', default=2)
    parser.add_argument('--data_saving_path', help='Directory for saving the log files for this run')
    parser.add_argument('--batch_size_per_process',
                        help='Number of samples collected for each process at each iteration', default=1000)
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=150)
    parser.add_argument('--run_dir', help='Directory for saving the log files for the large run')

    args = parser.parse_args()

    print(args)
    # batch_size_per_process = sys.argv[5]
    # num_iterations = sys.argv[6]
    main(args.env, args.seed, int(args.curr_run), args.data_saving_path, int(args.batch_size_per_process),
         int(args.num_iterations))
