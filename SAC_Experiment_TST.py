from rllab.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import os
import time
import numpy as np
from sac.algos import SAC
from sac.misc.plotter import QFPolicyPlotter
from sac.policies import GMMPolicy, LatentSpacePolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction, NNVFunction
cwd = os.getcwd()
print(cwd)
def run_task(*_):
    env = normalize(GymEnv('SimplerPathFinding-v0',record_video=False,force_reset=True))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    pool = SimpleReplayBuffer(max_replay_buffer_size=1e6, env_spec=env.spec)
    sampler = SimpleSampler(
        max_path_length=30, min_pool_size=100, batch_size=64)

    base_kwargs = dict(
        sampler=sampler,
        epoch_length=500,
        n_epochs=1000,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=24,
        eval_deterministic=False
    )

    M = 64
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M,M]
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M,M]
    )

    policy = GMMPolicy(
        env_spec=env.spec,
        K=4,
        hidden_layer_sizes=[M, M,M],
        qf=qf,
        reg=0.001
    )


    # plotter = QFPolicyPlotter(
    #     qf=qf,
    #     policy=policy,
    #     obs_lst=np.array([[-2.5, 0.0],
    #                       [0.0, 0.0],
    #                       [2.5, 2.5]]),
    #     default_action=[np.nan, np.nan],
    #     n_samples=100
    # )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy = policy,
        pool=pool,
        qf1=qf,
        qf2=qf,

        vf=vf,
        # plotter=plotter,
        lr=3e-4,
        # scale_reward=3.0,
        discount=0.99,
        # tau=1e-4,x

        save_full_state=True
    )
    algorithm.train()


# run_task()
timeStamp = time.time()
run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    log_dir=cwd+'/data/'+'Exp'+str(timeStamp),
    script=cwd+'/scripts/run_experiment_lite.py',
    seed=0,
    # plot=True,
)