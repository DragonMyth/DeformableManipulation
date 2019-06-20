from Util.post_training_process import *



def render_simple_particle_carving():
    # render_policy('SimplerParticleCarving-v0', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v0', stoch=False, record=True,
                   num_runs=1, policy_func=cnn_policy_fn)


def render_simple_particle_carving_rotation():
    # render_policy('SimplerParticleCarving-v1', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v1', stoch=False, record=True,
                  num_runs=1, policy_func=cnn_policy_fn)


def render_simple_particle_carving_circle_matching():
    # render_policy('SimplerParticleCarving-v2', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v2', stoch=False, record=True,
                  num_runs=1, policy_func=cnn_template_policy_fn)


def render_simple_particle_carving_circle_matching_explicit_circle():
    # render_policy('SimplerParticleCarving-v2', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v2', stoch=False, record=True,
                  num_runs=1, policy_func=cnn_explicity_target_policy_fn, random_policy=False,save_path=True,save_filename="data/trajs/simple_particle_traj.pkl")


def render_flex_granular_sweep():
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_explicit_target_policy_fn,save_filename="data/trajs/curr_1_granular.pkl")
    # render_policy("FlexGranularSweep-v0",stoch=False,record=False,random_policy=False,save_path=True,policy_func=cnn_granular_sweep_policy_fn,save_filename="data/trajs/no_target_granular.pkl")

    #file_name = "data/trajs/granular_voxel_bar_two_goal.pkl"
    #file_name = "data/trajs/granular_voxel_bar_goal_density.pkl"
    file_name = "data/trajs/granular_voxel_bar_goal_density_diff.pkl"

    render_policy("FlexGranularSweep-v1",stoch=False,record=False,random_policy=True,save_path=True,policy_func=cnn_granular_sweep_voxel_bar_policy_fn,save_filename=file_name)

# render_reacher()
# render_simple_particle_carving()
# render_simple_particle_carving_rotation()
# render_simple_particle_carving_circle_matching_explicit_circle()

def render_flex_goo_sweep():
    render_policy("FlexGooSweep-v0",stoch=False,record=False,random_policy=True,save_path=True,policy_func=cnn_granular_sweep_policy_fn)

render_flex_granular_sweep()
# render_flex_goo_sweep()