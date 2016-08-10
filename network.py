import os
os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'

import theano as th
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano import shared
import multiprocessing as mp
import ctypes
import pickle
import time

from theano.compile.nanguardmode import NanGuardMode
import numpy as np

from parameters import constants

def prodliste(liste):
    prod = 1
    for i in liste:
        prod *= i

    return prod

class ConvLayer:
    def __init__(self, inputs, filter_shape, stride, name_prefix, input_shape):
        self.inputs = inputs
        self.W = T.dtensor4(name = name_prefix + "_W")
        self.b = T.dvector(name = name_prefix  + "_b")

        conv_out = conv2d(
                    input = inputs,
                    input_shape  = input_shape,
                    filters      = self.W,
                    filter_shape = filter_shape,
                    subsample    = stride
                )


        addition = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.out = T.nnet.relu(addition)
        self.params = [self.W, self.b]

class ConvLayerVars:
    def __init__(self, rng, filter_shape, name=None, meansquare = False):
        self.name = name

        self.W = mp.RawArray(ctypes.c_double, prodliste(filter_shape))

        w = np.frombuffer(self.W).reshape(filter_shape)
        np.copyto(w, np.asarray(rng.normal(size=filter_shape, scale=constants.weigthInitStdev)))
        
        self.b = mp.RawArray(ctypes.c_double, filter_shape[0])
        
        b = np.frombuffer(self.b)
        b[:] = 1

        self.filter_shape = filter_shape

        self.mean_square = meansquare 
        if meansquare:
            self.W_ms  = mp.RawArray(ctypes.c_double, prodliste(filter_shape))
            w_ms = np.frombuffer(self.W_ms)
            w_ms[:] = 1

            self.b_ms = mp.RawArray(ctypes.c_double, filter_shape[0])
            b_ms = np.frombuffer(self.b_ms)
            b_ms[:] = 1

    def update_weights(self, conv):
        np.copyto(self.npweights(), conv.npweights())
        np.copyto(self.npbiases(),  conv.npbiases())

    def npweights(self):
        return np.frombuffer(self.W).reshape(self.filter_shape)

    def npbiases(self):
        return np.frombuffer(self.b)
        
    def npmsweights(self):
        return np.frombuffer(self.W_ms).reshape(self.filter_shape) if self.mean_square else None

    def npmsbiases(self):
        return np.frombuffer(self.b_ms) if self.mean_square else None

class FullyConectedLayerVars:
    def __init__(self, rng, nb_inputs, nb_outputs, name=None, meansquare=False):
        self.name = name
        self.shape = [nb_inputs, nb_outputs]
        self.W     = mp.RawArray(ctypes.c_double, nb_inputs * nb_outputs)

        w = np.frombuffer(self.W).reshape(self.shape)
        np.copyto(w, np.asarray(rng.normal(size=self.shape, scale=constants.weigthInitStdev)))

        self.b    = mp.RawArray(ctypes.c_double, nb_outputs)

        b = np.frombuffer(self.b)
        b[:] = 1

        self.mean_square = meansquare

        if meansquare:
            self.W_ms  = mp.RawArray(ctypes.c_double, nb_inputs * nb_outputs)
            w_ms = np.frombuffer(self.W_ms)
            w_ms[:] = 1

            self.b_ms = mp.RawArray(ctypes.c_double, nb_outputs)
            b_ms = np.frombuffer(self.b_ms)
            b_ms[:] = 1

    def update_weights(self, fcl):
        np.copyto(self.npweights(), fcl.npweights())
        np.copyto(self.npbiases(),  fcl.npbiases())

    def npweights(self):
        return np.frombuffer(self.W).reshape(self.shape)

    def npbiases(self):
        return np.frombuffer(self.b)

    def npmsweights(self):
        return np.frombuffer(self.W_ms).reshape(self.shape) if self.mean_square else None

    def npmsbiases(self):
        return np.frombuffer(self.b_ms) if self.mean_square else None


class FullyConectedLayer:
    def __init__(self, inputs,  activation, name_prefix):
        self.inputs = inputs

        self.W = T.dmatrix(name = name_prefix + "_W")
        self.b = T.dvector(name = name_prefix + "_b")
        
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]

class DeepQNet:
    def __init__(self, n_sorties, prefix, mean_square_buffer):
        rng = np.random.RandomState(42)
        self.prefix = prefix
        self.mutex = mp.Lock()
        self.conv1_hold = ConvLayerVars(rng, 
                                constants.conv1_shape,
                                prefix +"_conv1",
                                mean_square_buffer
                            )

        self.conv2_hold = ConvLayerVars(rng, 
                                constants.conv2_shape,
                                prefix + "_conv2",
                                mean_square_buffer
                            )

        self.fcl1_hold  = FullyConectedLayerVars(rng, 
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                prefix + "_fcl1",
                                mean_square_buffer
                            )

        self.fcl2_hold  = FullyConectedLayerVars(rng,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                prefix + "_fcl2",
                                mean_square_buffer
                            )

        
        self.holders = [
                        self.conv1_hold, 
                        self.conv2_hold, 
                        self.fcl1_hold,
                        self.fcl2_hold
                       ]

        self.mean_square = mean_square_buffer

    def update_weights(self, dqn):
        for local, distant in zip(self.holders, dqn.holders):
            local.update_weights(distant)


    def instantiate(self, inputs, prefix):
        self.conv1 = ConvLayer(
                        inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                        prefix + "_" + self.prefix + "_conv1",
                        constants.image_shape
                    )

        self.conv2 = ConvLayer(
                        self.conv1.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                        prefix + "_" + self.prefix + "_conv2",
                        None
                    )

        self.fcl1  = FullyConectedLayer(
                         self.conv2.out.flatten(),
                         T.nnet.relu,
                         prefix + "_" + self.prefix + "_fcl1"
                    )

        self.fcl2  = FullyConectedLayer(
                         self.fcl1.output,
                         None,
                         prefix + "_" + self.prefix + "_fcl2"
                    )

        self.weight_parameters = []
        for holder in self.holders:
            self.weight_parameters.append(holder.npweights())
            self.weight_parameters.append(holder.npbiases())

        self.layers = [self.conv1, self.conv2, self.fcl1, self.fcl2]
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        if self.mean_square:
            self.meansquare_params = []
            for holder in self.holders:
                self.meansquare_params.append(holder.npmsweights())
                self.meansquare_params.append(holder.npmsbiases())
    
    def save(self, filename):
        array_dict = {}
        for layer in [self.conv1_hold, self.conv2_hold, self.fcl1_hold, self.fcl2_hold]:
            name = layer.name
            array_dict[name+"_w"] = layer.npweights()
            array_dict[name+"_b"] = layer.npbiases()

        np.savez(filename, array_dict)

        
class AgentComputation:
    def __init__(self, network, critic, prefix):
        self.inputs = T.dtensor4(name=prefix+"_input")

        network.instantiate(self.inputs, prefix)
        critic.instantiate(self.inputs, prefix)

        self.network = network
        self.critic  = critic
        self.n = 0

        self.initialisedRMSVals = True

        best_actions   = T.argmax(network.fcl2.output)
        critic_score   = T.max(critic.fcl2.output)
        
        inputsWithNet = network.params + [self.inputs]

        with network.mutex:
            self._getBestAction = th.function(inputsWithNet, 
                                    [best_actions], 
                                    name="getBestAction")

        inputsWithCritic = critic.params + [self.inputs]
        with network.mutex:
            self._getCriticScore = th.function(inputsWithCritic, [critic_score], name = "getCriticScore")

        #Learning inputs
        self.actions = T.iscalar(prefix+'_actionLearn');
        self.labels  = T.dscalar(prefix+'_label')

        actions_score = network.fcl2.output[self.actions]
        error = .5 * (actions_score - self.labels)**2

        gradients  = [T.grad(error, param)   for param in network.params] 
        self.gradientsAcc  = [np.zeros_like(param) for param in network.weight_parameters]

        inputsWithNet = network.params + [self.inputs, self.actions, self.labels]

        with network.mutex:
            self._computeGradient = th.function(inputsWithNet, gradients, name = "computeGradients")

    def update_critic(self):
        self.critic.update_weights(self.network)

    def cumulateGradient(self, inputs, actions, labels, ident):
        gradients = self._computeGradient(*self.network.weight_parameters, inputs, actions, labels)
        for accumulator, gradient in zip(self.gradientsAcc, gradients):
            np.add(accumulator, gradient, accumulator)
        self.n += 1

    def getBestAction(self, inputs):
        return self._getBestAction(*self.network.weight_parameters, inputs)

    def getCriticScore(self, inputs):
        return self._getCriticScore(*self.critic.weight_parameters, inputs)

    def applyGradient(self, learning_rate):
        if self.n == 0:
            return
        #Meansquare value of gradient updates
        if self.initialisedRMSVals:
            for ms, accumulator in zip(self.network.meansquare_params, self.gradientsAcc):
                np.multiply(ms, constants.decay_factor, ms)
                B = np.square(accumulator)
                np.multiply(B, 1-constants.decay_factor, B)
                np.add(ms, B, ms)
        else :
            self.initialisedRMSVals = True
            for ms, accumulator in zip(self.network.meansquare_params, self.gradientsAcc):
                np.copyto(ms, accumulator / self.n)
            

        #Parameter updates
        #i = 0
        for param, accumulator, ms in zip(  self.network.weight_parameters, 
                                            self.gradientsAcc, 
                                            self.network.meansquare_params
                                         ):
            #print(i, "Before : ", str(np.sum(param)))
            G = np.sqrt(ms + constants.epsilon_cancel)  
            np.multiply(G, self.n, G)
            np.divide(accumulator, G, accumulator)
            np.multiply(accumulator, learning_rate, accumulator)
            np.subtract(param, accumulator, param)
            accumulator.fill(0)
            #print(i, "After : ", str(np.sum(param)))
            #i+=1
        self.n = 0
