import theano as th
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano import shared
import multiprocessing as mp
import ctypes
import pickle

import numpy as np

from parameters import constants

def prodliste(liste):
    prod = 1
    for i in liste:
        prod *= i

    return prod

class ConvLayer:
    def __init__(self, convl, inputs, filter_shape, stride, name_prefix):
        self.inputs = inputs
        self.W = T.dtensor4()
        self.b = T.dvector()

        conv_out = conv2d(
                    input = inputs,
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
        np.copyto(w, np.asarray(rng.normal(size=filter_shape)))
        
        self.b = mp.RawArray(ctypes.c_double, filter_shape[0])
        
        b = np.frombuffer(self.b)
        b[:] = 0.01

        self.filter_shape = filter_shape

        self.mean_square = meansquare 
        if meansquare:
            self.W_ms  = mp.RawArray(ctypes.c_double, prodliste(filter_shape))
            w_ms = np.frombuffer(self.W)
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
        np.copyto(w, np.asarray(rng.normal(size=self.shape)))

        self.b    = mp.RawArray(ctypes.c_double, nb_outputs)

        b = np.frombuffer(self.b)
        b[:] = 0.01

        self.mean_square = meansquare

        if meansquare:
            self.W_ms  = mp.RawArray(ctypes.c_double, nb_inputs * nb_outputs)
            w_ms = np.frombuffer(self.W)
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
    def __init__(self, inputs,  fcl, activation, name_prefix):
        self.inputs = inputs

        self.W = T.dmatrix()
        self.b = T.dvector()
        
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]

class DeepQNet:
    def __init__(self, n_sorties, prefix, mean_square_buffer):
        rng = np.random.RandomState(42)
        self.prefix = prefix
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

        self.mean_square = mean_square

    def update_weights(self, dqn):
        for local, distant in zip(self.holders, dqn.holders):
            local.update_weights(distant)


    def instantiate(self, inputs, prefix):
        self.conv1 = ConvLayer(self.conv1_hold, 
                        inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                        prefix + "_" + self.prefix + "_conv1"
                    )

        self.conv2 = ConvLayer(self.conv2_hold, 
                        self.conv1.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                        prefix + "_" + self.prefix + "_conv2"
                    )

        self.fcl1  = FullyConectedLayer(
                         self.conv2.out.flatten(ndim=2),
                         self.fcl1_hold,
                         T.nnet.relu,
                         prefix + "_" + self.prefix + "_fcl1"
                    )

        self.fcl2  = FullyConectedLayer(
                         self.fcl1.output,
                         self.fcl2_hold,
                         None,
                         prefix + "_" + self.prefix + "_fcl2"
                    )

        self.weight_parameters = []
        for holder in self.holders:
            self.weight_parameters += holder.npweights()
            self.weight_parameters += holder.npbiases()

        self.layers = [self.conv1, self.conv2, self.fcl1, self.fcl2]
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        if self.mean_square:
            self.meansquare_params = []
            for holder in self.holders:
                self.meansquare_params += holder.npmsweights()
                self.meansquare_params += holder.npmsbiases()
    
    def wrap_function(self, inputs, outputs):
        complete_inputs = self.params + inputs
        f = th.function(complete_inputs, outputs)
        return lambda x : f(self.weight_parameters + x)

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

        updatable = network.layers

        params = []
        for layer in updatable:
            params            += layer.params
        
        best_actions   = T.argmax(network.fcl2.output)
        critic_score   = T.max(critic.fcl2.output)

        self.getBestAction  = network.wrap_function([self.inputs], [best_actions])
        self.getCriticScore = network.wrap_function([self.inputs], [critic_score])

        #Learning inputs
        self.actions = T.ivector(prefix+'_actionsVector');
        self.labels  = T.dvector(prefix+'_labels')

        actions_scores = network.fcl2.output[T.arange(self.actions.shape[0]), self.actions]
        error = T.mean(.5 * (actions_scores - self.labels)**2)

        gradients  = [T.grad(error, param)   for param in params] 

        self.gradientsAcc = [np.zeros(param.shape)) for param in network.weight_parameters]

        self.computeGradient = network.wrap_function([self.inputs, self.actions, self.labels],
                                    [gradients])

    def self.update_critic(self):
        self.critic.update_weights(self.network)

    def cumulateGradient(self, inputs, actions, labels):
        gradients = self.computeGradient(inputs, actions, labels)
        for accumulator, gradient in zip(self.gradientsAcc, gradients):
            accumulator += gradient

    def applyGradient(self, learning_rate):
        #Meansquare value of gradient updates
        for ms, accumulator in zip(self.network.meansquare_params, self.gradientsAcc):
            np.multiply(ms, constants.decay_factor, ms)
            B = np.sqr(accumulator)
            np.multiply(B, 1-constants.decay_factor, B)
            np.add(ms, B, ms)

        #Parameter updates
        for param, accumulator, ms in zip(  self.network.weight_parameters, 
                                            self.gradientsAcc, 
                                            self.network.meansquare_params
                                         ):
            G = np.sqrt(ms + epsilon_cancel)  
            np.divide(accumulator, G, accumulator)
            np.substract(param, accumulator, param)
            accumulator.fill(0)
