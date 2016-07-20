class constants:
    input_frames    = 4
    input_size      = 84*84
    conv1_nbfilt    = 32
    image_shape     = [None, 1, 84, 84,]
    conv1_shape     = (16,1,8, 8)
    conv1_zwidth    = 16
    conv1_strides   = (4, 4)
    conv2_shape     = [32, 16, 4, 4]
    conv2_zwidth    = 32
    conv2_strides   = (2, 2)
    cnn_output_size = conv2_zwidth * 9 * 9
    fcl1_nbUnit     = 256
    max_noop        = 30
    final_e_frame   = 1000000
    action_repeat   = 4
    discount_factor = 0.99
    decay_factor    = 0.95
    nb_thread       = 20
    nb_max_frames   = 80000000
    batch_size      = 5
    critic_up_freq  = 40000
    epsilon_cancel  = 0.00001

class shared:
    nb_actions      = 0
    game_name       = ''
    T               = 0 
