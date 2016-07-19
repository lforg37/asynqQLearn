from threading import Thread
from parameters import shared, constants
from ale_python_interface import ALEInterface
from random import random, randrange
from network import AgentComputation
from improc import NearestNeighboorInterpolator2D

import numpy as np

class AgentThread(Thread):
    def __init__(self, lock, globalNet, game_path, ident):
        Thread.__init__(self)
        self.lock = lock
        self.ale = ALEInterface()
        self.id = ident
        self.ale.setInt(b'random_seed', randrange(0,256,1))
        self.ale.loadROM(game_path)
        self.actions = self.ale.getMinimalActionSet()

        self.globalNet = globalNet
        self.computation  = AgentComputation(globalNet)

        self.t = 0
        self.scores = []
        rnd = random()
        if   rnd < 0.4:
            self.epsilon_end = 0.1
        elif rnd < 0.7:
            self.epsilon_end = 0.01
        else:
            self.epsilon_end = 0.5
        
    def run(self): 
        interpolator = NearestNeighboorInterpolator2D([210,160],[84,84])
        images = np.empty([constants.action_repeat, 84, 84, 1], dtype=np.uint8) 

        current_frame = np.empty([210, 160, 1], dtype=np.uint8)
        self.ale.getScreenGrayscale(current_frame)

        state = interpolator.interpolate(current_frame)

        reward_batch = []
        action_batch = []
        state_batch  = []
        
        score = 0

        with self.lock:
            T = shared.T
            shared.T += 1

        while T < constants.nb_max_frames:
            state_batch.append(state.transpose(2,0,1))
            epsilon = self.epsilon_end
            if T < constants.final_e_frame:
                epsilon = 1 + (self.epsilon_end - 1) * T / constants.final_e_frame

            rnd = random()
            action = randrange(0,len(self.actions)) if rnd < epsilon else self.globalNet.getBestAction(state.transpose(2,0,1)[np.newaxis])[0]
            action_batch.append(action)
            

            reward = 0
            i = 0
            while i < constants.action_repeat and not self.ale.game_over():
                reward += self.ale.act(self.actions[action])
                self.ale.getScreenGrayscale(current_frame)
                images[i] = interpolator.interpolate(current_frame)
                i += 1

            state = np.maximum.reduce(images[0:i], axis=0)

            discounted_reward = 0 if self.ale.game_over() else self.globalNet.getCriticScore(state.transpose(2,0,1)[np.newaxis])[0]
            

            score += reward
            reward_batch.append(reward + constants.discount_factor * discounted_reward)

            if self.t % constants.batch_size == 0 or self.ale.game_over():
                self.computation.cumulateGradient(np.asarray(state_batch), np.asarray(action_batch, dtype=np.int32), np.asarray(reward_batch))
                reward_batch = []
                action_batch = []
                state_batch  = []
            
            if T % constants.critic_up_freq == 0:
                print("Update critic !")
                self.globalNet.update_critic()
            
            if self.ale.game_over():
                print("["+str(self.id)+"] Game ended with score of : "+str(score))
                self.ale.reset_game()
                self.scores.append(score)
                if len(self.scores) >= 12:
                    moy = sum(self.scores) / len(self.scores)
                    print("Average scores for last 12 games for thread "+str(self.id)+ " : " + str(moy))
                    self.scores = []
                score = 0

            with self.lock: 
                T = shared.T
                shared.T += 1

            self.t += 1
