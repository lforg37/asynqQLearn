class constants:
    input_frames     = 4
    input_size       = 84*84
    conv1_nbfilt     = 32
    image_shape      = [4, 1, 84, 84,]
    conv1_shape      = (16,1,8, 8)
    conv1_zwidth     = 16
    conv1_strides    = (4, 4)
    conv2_shape      = [32, 16, 4, 4]
    conv2_zwidth     = 32
    conv2_strides    = (2, 2)
    cnn_output_size  = conv2_zwidth * 9 * 9 * 4
    fcl1_nbUnit      = 256
    max_noop         = 30
    final_e_frame    = 4000000
    action_repeat    = 4
    discount_factor  = 0.99
    decay_factor     = 0.99
    nb_agent         = 1
    nb_max_frames    = 80000000
    batch_size       = 5
    critic_up_freq   = 40000
    epsilon_cancel   = 1
    weigthInitStdev  = 0.25
    level_error      = 0.000000001

    filebase = 'output_agent_'
    
    
    lock_T      = True
    lock_read   = True
    lock_write  = True
    lenmoy      = 12
