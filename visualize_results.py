from Util.post_training_process import *


#
# def render_reacher():
#     # render_policy('SimplerParticleCarving-v0', stoch=True, record=True,
#     #               autoencoder_name_list=[], num_runs=1,random_policy=True)
#     render_policy('DartReacher3d-v0', stoch=False, record=True,
#                   autoencoder_name_list=[], num_runs=1, policy_func=cnn_policy_fn)

def render_simple_particle_carving():
    # render_policy('SimplerParticleCarving-v0', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v0', stoch=False, record=True,
                  autoencoder_name_list=[], num_runs=1, policy_func=cnn_policy_fn)


def render_simple_particle_carving_rotation():
    # render_policy('SimplerParticleCarving-v1', stoch=True, record=True,
    #               autoencoder_name_list=[], num_runs=1, random_policy=True)
    render_policy('SimplerParticleCarving-v1', stoch=False, record=True,
                  autoencoder_name_list=[], num_runs=1, policy_func=cnn_policy_fn)


# render_reacher()
# render_simple_particle_carving()
render_simple_particle_carving_rotation()
