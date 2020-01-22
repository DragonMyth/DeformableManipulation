from Util.post_training_process import *


def render_flex_granular_sweep():
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_explicit_target_policy_fn,save_filename="data/trajs/curr_1_granular.pkl")
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_policy_fn,save_filename="data/trajs/no_target_granular.pkl")

    #file_name = "data/trajs/granular_voxel_bar_two_goal.pkl"
    #file_name = "data/trajs/granular_voxel_bar_goal_density.pkl"

    #file_name = "data/trajs/granular_rot_base.pkl"
    # file_name = "data/trajs/granular_controllable_ghost_one_goal.pkl"
    file_name = "data/trajs/plasticOneClusterBarCentric.pkl"
    # flex_fc_policy_fn
    # cnn_granular_sweep_voxel_bar_policy_fn
    # ppo_FlexGranularSweep - v4
    # FlexPlasticReshaping - v6
    render_policy("FlexPlasticReshaping-v6",stoch=True,record=True,random_policy=False,save_path=True,policy_func=cnn_cnn_particle_sweep_multi_img_policy_share_val_fn,save_filename=file_name)
    # render_policy("FlexPlasticReshaping-v7",stoch=False,record=True,random_policy=False,save_path=True,policy_func=cnn_cnn_particle_sweep_multi_img_policy_share_val_fn,save_filename=file_name)


def render_flex_plastic_reshaping():
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_explicit_target_policy_fn,save_filename="data/trajs/curr_1_granular.pkl")
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_policy_fn,save_filename="data/trajs/no_target_granular.pkl")

    #file_name = "data/trajs/granular_voxel_bar_two_goal.pkl"
    #file_name = "data/trajs/granular_voxel_bar_goal_density.pkl"
    #file_name = "data/trajs/granular_rot_base.pkl"
    # file_name = "data/trajs/granular_controllable_ghost_one_goal.pkl"
    file_name = "data/trajs/dummy.pkl"

    render_policy("FlexPlasticReshaping-v1",stoch=False,record=False,random_policy=True,save_path=True,policy_func=flex_fc_policy_fn,save_filename=file_name)


render_flex_granular_sweep()
# render_flex_plastic_reshaping()
# render_flex_goo_sweep()