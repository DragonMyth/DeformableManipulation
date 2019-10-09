from spinup import sac
import gym
import tensorflow as tf
env_fn = lambda : gym.make("DartCartPole-v1")


logger_kwargs = dict(output_dir='spinupDeform/data', exp_name='testSacHopper')
ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.tanh)
sac(env_fn, ac_kwargs=ac_kwargs, seed=0,start_steps=0,logger_kwargs=logger_kwargs)