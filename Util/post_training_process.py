import random

from matplotlib import cm
from mpi4py import MPI

import gym
import joblib
from joblib import Parallel, delayed
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import tkinter
from tkinter.filedialog import askdirectory
import numpy as np

import matplotlib.pyplot as plot
from boltons import iterutils
import multiprocessing
import matplotlib.colors
import os.path as osp

from baselines import bench as bc
from baselines import logger

import tensorflow as tf
from baselines.ppo1 import mlp_policy, cnn_policy_carving, \
    cnn_policy_carving_two_maps,cnn_policy_particle_sweep,mlp_policy_flex
import baselines.common.tf_util as U

import itertools
from keras.models import load_model
from gym import wrappers
import seaborn as sns


# def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
#     return mlp_policy_novelty.MlpPolicyNovelty(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64,
#                                                num_hid_layers=3,

#                                                )
def cnn_particle_sweep_policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    # return cnn_policy_carving.CnnPolicyCarving(name=name, ob_space=ob_space, ac_space=ac_space)
    return cnn_policy_particle_sweep.CnnPolicyParticleSweep(name=name, ob_space=ob_space,
                                                                    ac_space=ac_space)

def cnn_template_policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return cnn_policy_carving_two_maps.CnnPolicyCarvingTwoMaps(name=name, ob_space=ob_space, ac_space=ac_space
                                                               )


def cnn_policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return cnn_policy_carving.CnnPolicyCarving(name=name, ob_space=ob_space, ac_space=ac_space
                                               )


def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=256,
                                num_hid_layers=2,
                                )
def flex_fc_policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return mlp_policy_flex.MlpPolicyFlex(name=name, ob_space=ob_space, ac_space=ac_space
                                )


def perform_rollout(policy,
                    env,
                    snapshot_dir=None,
                    animate=False,
                    plot_result=False,
                    control_step_skip=1,
                    costum_horizon=None,
                    stochastic=True,
                    debug=False,
                    saved_rollout_path=None,
                    num_repeat=1,
                    record=False
                    ):

    predefined_actions = None
    if (saved_rollout_path):
        predefined_actions = joblib.load(saved_rollout_path)['actions']
    # print(predefined_actions)
    all_reward_info = []
    env.seed(42)
    for _ in range(num_repeat):

        path = {'observations': [], 'actions': []}
        last_action = None

        horizon = env._max_episode_steps
        if costum_horizon != None:
            horizon = costum_horizon
        if animate:
            # env.env.env.
            from gym.wrappers.monitoring import Monitor
            env = Monitor(env,snapshot_dir+'/../policy_runs',force=True)
            if hasattr(env.unwrapped, 'disableViewer'):
                env.unwrapped.disableViewer = False
            if record:
                env.unwrapped.save_video = True
                env.unwrapped.video_path = snapshot_dir + '/../policy_runs'
                import os
                if not os.path.exists(env.unwrapped.video_path):
                    os.makedirs(env.unwrapped.video_path)

        observation = env.reset()

        for i in range(horizon):
            if i % 200 == 0 and debug:
                print("Current Timestep:", i)
            if animate:
                env.render()
            if policy is None:

                action_taken = np.zeros((env.unwrapped.numInstances, env.unwrapped.act_dim))
            else:
                if i % control_step_skip == 0:
                    action_taken = policy.act(stochastic, observation)[0]
                    last_action = action_taken
                else:
                    action_taken = last_action
            if animate:
                if (policy is not None):
                    log_std = policy.log_std()
                    # print(log_std)

            #         std = policy.log_std(observation)[0]
            if predefined_actions:
                action_taken = predefined_actions[i][::-1]
            # print("MAX: ",np.max(action_taken),"MIN: ", np.min(action_taken))
            observation, reward, done, info = env.step(action_taken)
            # observation, reward, done, info = env.step(np.array([-1, 0]))

            reward = reward[0] if type(reward) == tuple else reward
            path['observations'].append(observation)
            path['actions'].append(action_taken)
            all_reward_info.append(info)

            if done:
                # print("Rollout is Done")
                break

    if plot_result:
        iters = np.arange(1, len(all_reward_info) + 1, 1)
        data_list = {}
        for i in range(len(all_reward_info)):

            for key in all_reward_info[0]:
                if key not in data_list:
                    data_list[key] = []
                data_list[key].append(all_reward_info[i][key])

        random.seed = 41
        cnt = 0

        plot.figure()

        # total_ret = np.sum(data_list['rwd'])
        # print("Total return of the trajectory is: ", total_ret)

        # sns.set_palette('hls', len(data_list))
        for key in sorted(data_list.keys()):
            # print(key)
            if (key != 'actions' and key != 'states'):
                cnt += 1
                plot.plot(iters, data_list[key],
                          label=str(key))
                # plot.yscale('symlog')

        plot.xlabel('Time Steps')
        plot.ylabel('Step Reward')
        plot.legend()
        plot.savefig(snapshot_dir + '/rewardDecomposition.jpg')

        plot.figure()
        plot.show()

    return path

def render_policy(env_name, save_path=False, save_filename="data/trajs/test_path_name.pkl",
                  stoch=False,
                  record=False, policy_func=policy_fn, random_policy=False, num_runs=1):
    if not random_policy:
        root = tkinter.Tk()
        # root.update()
        openFileOption = {}
        openFileOption['initialdir'] = '../data/ppo_' + env_name
        filenames = askopenfilenames(**openFileOption)
        root.update()

        env = gym.make(env_name)

        for i in range(len(filenames)):
            filename = filenames[i]
            print(filename)
            policy_param = joblib.load(filename)

            snapshot_dir = filename[0:filename.rfind('/') + 1]

            # env = gym.wrappers.Monitor(env, snapshot_dir, force=True)
            tf.reset_default_graph()

            with U.single_threaded_session() as sess:
                # env = gym.make(env_name)

                current_palette = sns.color_palette()
                color = current_palette[i]
                env.env.trail_color = color

                pi = policy_func('pi', env.observation_space, env.action_space)

                restore_policy(sess, pi, policy_param)

                # summary_writer = tf.summary.FileWriter('./tflog', graph=sess.graph)
                if not stoch:
                    num_runs = 1
                for r in range(num_runs):
                    path = perform_rollout(pi, env, snapshot_dir=snapshot_dir, animate=True, plot_result=True,
                                           stochastic=stoch,
                                           saved_rollout_path=None,
                                           record=record,
                                           )
        if save_path:
            joblib.dump(path, save_filename)
    else:
        env = gym.make(env_name)

        path = perform_rollout(None, env, snapshot_dir='.', animate=True, plot_result=True,
                        stochastic=stoch,
                        saved_rollout_path=None,
                        record=False,
                        costum_horizon = 100
                        )
        if save_path:
            joblib.dump(path, save_filename)



def restore_policy(sess, policy, policy_params):
    cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    for i in range(len(policy.get_variables())):
        assign_op = policy.get_variables()[i].assign(
            policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)

