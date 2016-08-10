from parameters import constants
from ale_python_interface import ALEInterface
from random import random, randrange
import sys
from improc import NearestNeighboorInterpolator2D
from utils import LockManager
from network import AgentComputation
#from PIL import Image

import os

import numpy as np

def AgentProcess(rwlock, mainNet, criticNet, T_glob, T_lock, game_path, ident, init_learning_rate, barrier):

    #Assign processor cores to Agent
    os.system('taskset -p -c ' + str(ident) + ' ' + str(os.getpid()))

    #Set up game environment
    ale = ALEInterface()
    ale.setInt(b'random_seed', randrange(0,256,1))
    #ale.setBool(b'display_screen', True)
    ale.setBool(b'color_averaging', True)
    ale.loadROM(game_path)
    actions = ale.getMinimalActionSet()

    #Create Agent network based on shared weights wrapped in mainNet and criticNet
    computation = AgentComputation(mainNet, criticNet, 'computation_'+str(ident))

    f = open(constants.filebase+str(ident), 'w')

    t_lock      = LockManager(T_lock.acquire, T_lock.release, constants.lock_T)
    writer_lock = LockManager(rwlock.writer_acquire, rwlock.writer_release, constants.lock_write)
    reader_lock = LockManager(rwlock.reader_acquire, rwlock.reader_release, constants.lock_read)

    t = 0
    scores = []

    #Determination of Agent final epsilon-greedyness level
    rnd = random()
    if   rnd < 0.4:
        epsilon_end = 0.1
    elif rnd < 0.7:
        epsilon_end = 0.01
    else:
        epsilon_end = 0.5
    
    interpolator = NearestNeighboorInterpolator2D([210,160],[84,84])

    current_frame = np.empty([210, 160, 1], dtype=np.uint8)
    ale.getScreenGrayscale(current_frame)
    
    
    #Image.fromarray(current_frame.squeeze(),  mode='L').save('curframe.png')
    
    next_state    = np.empty([constants.action_repeat, 84, 84, 1], dtype=np.float32)
    #Image.fromarray(state[0].squeeze(), mode='L').save('prevgetcurframe.png')
    interpolator.interpolate(current_frame, next_state[0])
    next_state[1:4] = next_state[0]
    #Image.fromarray((state[0].squeeze()*255).astype(np.uint8), mode='L').save('getcurframe.png')

    score = 0

    if ident == 0:
        computation.initialisedRMSVals = False
    
    with t_lock:
        T = T_glob.value
        T_glob.value += 1

    barrier.wait()
    f.write("After the barrier !")
    f.flush()

    while T < constants.nb_max_frames:
        state      = next_state
        next_state = np.empty_like(state)

        # Determination of epsilon for the current frame
        # Epsilon linearlily decrease from one to self.epsilon_end
        # between frame 0 and frame constants.final_e_frame
        epsilon = epsilon_end
        if T < constants.final_e_frame:
            epsilon = 1 + (epsilon_end - 1) * T / constants.final_e_frame

        #Choosing current action based on epsilon greedy behaviour
        rnd = random()
        if rnd < epsilon:
            action = randrange(0, len(actions))
        else:
            with reader_lock:
                action = computation.getBestAction(state.transpose(0,3,1,2))[0]

        t += 1
 
        reward = 0

        i      = 0

        #repeating constants.action_repeat times the same action 
        #and cumulating the rewards 
        while i < constants.action_repeat and not ale.game_over():
            reward += ale.act(actions[action])
            ale.getScreenGrayscale(current_frame)
            interpolator.interpolate(current_frame, next_state[i])
            i += 1

        while i < constants.action_repeat:
            next_state[i] = next_state[i-1]
            i += 1

        #for i in range(constants.action_repeat):
        #    im = Image.fromarray((next_state[i].squeeze()*255).astype(np.uint8), mode='L')
        #    im.save(constants.filebase+str(ident)+'_image_'+'{:08d}'.format(T)+'_'+str(i)+'.png')

        score += reward
        
        discounted_reward = 0
        if   reward > 0:
            discounted_reward = 1
        elif reward < 0:
            discounted_reward = -1


        if not ale.game_over():
            #Computing the estimated Q value of the new state
            with reader_lock:
                discounted_reward += constants.discount_factor * computation.getCriticScore(next_state.transpose(0,3,1,2))[0]

        computation.cumulateGradient(
                    state.transpose(0,3,1,2), 
                    action, 
                    discounted_reward, ident)

        if t != 0 and (t % constants.batch_size == 0 or ale.game_over()):
            #computing learning rate for current frame
            lr = init_learning_rate# * (1 - T/constants.nb_max_frames)
            with writer_lock:
                computation.applyGradient(lr)
            t = 0

        if T % constants.critic_up_freq == 0:
            f.write("Update critic !\n")
            f.flush()
            with writer_lock:
                computation.update_critic()
            
        #Log some statistics about played games
        if ale.game_over():
            f.write("["+str(ident)+"] Game ended with score of : "+str(score) + "\n")
            f.write("["+str(ident)+"] T : "+str(T)+"\n")
            ale.reset_game()
            interpolator.interpolate(current_frame, next_state[0])
            next_state[1:4] = next_state[0]
            scores.append(score)
            if len(scores) >= constants.lenmoy:
                moy = sum(scores) / len(scores)
                f.write("Average scores for last 12 games for Agent "+str(ident)+ " : " + str(moy)+"\n")
                f.flush()
                scores = []
            score = 0

        with t_lock:
            T = T_glob.value
            T_glob.value += 1

