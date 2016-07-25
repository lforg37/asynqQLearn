from parameters import shared, constants
from ale_python_interface import ALEInterface
from random import random, randrange
from network import AgentComputation
import sys
from improc import NearestNeighboorInterpolator2D

import numpy as np

#Return value sampled from logUniform(10-4, 10-2) 
def logUniform():
    rnd = random()
    logsample = -4 + 2*rnd
    return 10 ** logsample
    

def AgentProcess(rwlock, globalNet, T_glob, T_lock, game_path, ident):
    ale = ALEInterface()
    ale.setInt(b'random_seed', randrange(0,256,1))
    ale.loadROM(game_path)
    actions = ale.getMinimalActionSet()

    computation = AgentComputation(globalNet, 'computation_'+str(ident))

    init_learning_rate = logUniform()

    f = open('output_thread_'+str(ident), 'w')

    t = 0
    scores = []
    rnd = random()
    if   rnd < 0.4:
        epsilon_end = 0.1
    elif rnd < 0.7:
        epsilon_end = 0.01
    else:
        epsilon_end = 0.5
    
    interpolator = NearestNeighboorInterpolator2D([210,160],[84,84])
    images = np.empty([constants.action_repeat, 84, 84, 1], dtype=np.uint8) 

    current_frame = np.empty([210, 160, 1], dtype=np.uint8)
    ale.getScreenGrayscale(current_frame)

    state = interpolator.interpolate(current_frame)

    score = 0
    
    with T_lock:
        T = T_glob
        T_glob += 1

    while T < constants.nb_max_frames:
        old_state = state
        epsilon = epsilon_end
        if T < constants.final_e_frame:
            epsilon = 1 + (epsilon_end - 1) * T / constants.final_e_frame

        rnd = random()
        if rnd < epsilon:
            action = randrange(0, len(actions))
        else:
            rwlock.reader_acquire() 
            action = computation.getBestAction(state.transpose(2,0,1)[np.newaxis])[0]
            rwlock.reader_release()
        self.t += 1
            
        reward = 0
        i      = 0
        while i < constants.action_repeat and not ale.game_over():
            reward += ale.act(actions[action])
            ale.getScreenGrayscale(current_frame)
            images[i] = interpolator.interpolate(current_frame)
            i += 1

        state = np.maximum.reduce(images[0:i], axis=0)
        
        if ale.game_over():
            discounted_reward = 0
        else:
            rwlock.reader_acquire()
            discounted_reward = computation.getCriticScore(state.transpose(2,0,1)[np.newaxis])[0]
            rwlock.reader_release()

        score += reward
        computation.cumulateGradient(
                    np.asarray(old_state.transpose(2,0,1)[np.newaxis]), 
                    np.asarray(action, dtype=np.int32), 
                    np.asarray(discounted_reward))

        if t % constants.batch_size == 0 or ale.game_over():
            lr = init_learning_rate * (1 - T/nb_max_frames)
            rwlock.writer_acquire()
            computation.applyGradient(lr)
            rwlock.writer_release()
            self.t = 0

        if T % constants.critic_up_freq == 0:
            f.write("Update critic !\n")
            f.flush()
            rwlock.writer_acquire()
            globalNet.update_critic()
            rwlock.writer_release()
            
        if self.ale.game_over():
            f.write("["+str(ident)+"] Game ended with score of : "+str(score) + "\n")
            f.flush()
            ale.reset_game()
            scores.append(score)
            if len(scores) >= 12:
                moy = sum(scores) / len(scores)
                f.write("Average scores for last 12 games for thread "+str(ident)+ " : " + str(moy)+"\n")
                f.flush()
                scores = []
            score = 0

        with T_lock:
            T = T_glob
            T_glob += 1

