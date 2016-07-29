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
        self.W = shared(
                        convl.npweights(),
                        borrow = True,
                        name = name_prefix + "_w"
                    )
        self.b = shared(
                        convl.npbiases(),
                        borrow = True,
                        name = name_prefix + "_b"
                    )


        conv_out = conv2d(
                    input = inputs,
                    filters      = self.W,
                    filter_shape = filter_shape,
                    subsample    = stride
                )


        addition = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.out = T.nnet.relu(addition)
        self.params = [self.W, self.b]
        
        self.meansquare_param = []

        if convl.mean_square:
            self.W_ms = shared(
                            np.frombuffer(convl.W_ms).reshape(filter_shape),
                            borrow = True,
                            name = name_prefix+"_WMS"
                        )
            self.b_ms = shared(
                            np.frombuffer(convl.b_ms),
                            borrow = True, 
                            name = name_prefix+"_bMS"
                        )
            self.meansquare_param = [self.W_ms, self.b_ms]


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

    def npweights(self):
        return np.frombuffer(self.W).reshape(self.filter_shape)

    def npbiases(self):
        return np.frombuffer(self.b)
        

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

    def npweights(self):
        return np.frombuffer(self.W).reshape(self.shape)

    def npbiases(self):
        return np.frombuffer(self.b)


class FullyConectedLayer:
    def __init__(self, inputs,  fcl, activation, name_prefix):
        self.inputs = inputs

        self.W = shared(
                        fcl.npweights(),
                        borrow = True,
                        name = name_prefix + "_w"
                    )
        self.b = shared(
                        fcl.npbiases(),
                        borrow = True,
                        name = name_prefix + "_b"
                    )
        
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]

        self.meansquare_param = []

        if fcl.mean_square:
            self.W_ms = shared(
                            np.frombuffer(fcl.W_ms).reshape(fcl.shape),
                            borrow = True,
                            name = name_prefix+"_WMS"
                        )
            self.b_ms = shared(
                            np.frombuffer(fcl.b_ms),
                            borrow = True, 
                            name = name_prefix+"_bMS"
                        )
            self.meansquare_param = [self.W_ms, self.b_ms]

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

        self.layers = [self.conv1, self.conv2, self.fcl1, self.fcl2]

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

        critic_updates = []
        for actor_layer, critic_layer in zip(network.layers, critic.layers):
            critic_updates.append((critic_layer.W, actor_layer.W))
            critic_updates.append((critic_layer.b, actor_layer.b))

        self.update_critic = th.function([],[], updates=critic_updates)

        updatable = network.layers

        params = []
        meansquare_params = []
        for layer in updatable:
            params            += layer.params
            meansquare_params += layer.meansquare_param
        
        best_actions        = T.argmax(network.fcl2.output)
        critic_score   = T.max(critic.fcl2.output)

        self.getBestAction  = th.function([self.inputs],[best_actions])
        self.getCriticScore = th.function([self.inputs],[critic_score])

        #Learning inputs
        self.actions = T.ivector(prefix+'_actionsVector');
        self.labels  = T.dvector(prefix+'_labels')
        self.learning_rate = T.dscalar()

        actions_scores = network.fcl2.output[T.arange(self.actions.shape[0]), self.actions]
        error = T.mean(.5 * (actions_scores - self.labels)**2)

        gradients  = [T.grad(error, param)   for param in params] 

        self.gradientsAcc = [shared(np.zeros(param.get_value().shape)) for param in params]
        self.clearGradients()

        accGradUpdates = [(accGradient, accGradient + gradient) \
                            for accGradient, gradient in zip(self.gradientsAcc, gradients)]

        self.cumulateGradient = th.function([self.inputs, self.actions, self.labels],
                                    [],
                                    updates = accGradUpdates)

        #g <- \alpha g + (1-\alpha) d\theta^2
        meansquare_update = [(ms, constants.decay_factor * ms + (1-constants.decay_factor) * T.sqr(grad))\
                                for ms, grad in zip(meansquare_params, self.gradientsAcc)]

        #\theta <- \theta - \etha * d\theta/(g + \epsilon)^(1/2)
        param_update  =  [(param, 
            param - self.learning_rate * gradient / T.sqrt(square_mean + constants.epsilon_cancel)) \
                        for param, gradient, square_mean in zip(params, 
                                    self.gradientsAcc,
                                    meansquare_params
                                )]

        self.applyGradient = th.function([self.learning_rate],[],updates = meansquare_update + param_update)


    def clearGradients(self):
        for var in self.gradientsAcc:
            var.set_value(np.zeros(var.get_value().shape))
