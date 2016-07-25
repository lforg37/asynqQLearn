import theano
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano import shared
import multiprocessing as mp
import ctypes
from operator import mul

import numpy as np

from parameters import constants

class ConvLayer:
    def __init__(self, convl, inputs, filter_shape, stride, name_prefix):
        self.inputs = inputs
        self.W = shared(
                        np.frombuffer(convl.W.get_obj()).resize(filter_shape), 
                        borrow = True,
                        name = name_prefix^+ "_w"
                    )
        self.b = shared(
                        np.frombuffer(convl.b.get_obj()).resize(filter_shape[0]),
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
                            np.frombuffer(convl.W_ms.get_obj()).reshape(filter_shape),
                            borrow = True,
                            name = name_prefix+"_WMS"
                        )
            self.b_ms = shared(
                            npo.frombuffer(convl.b_ms.get_obj()),
                            borrow = True, 
                            name = name_prefix+"_WMS"
                        )
            self.meansquare_param = [self.W_ms, self.b_ms]


class ConvLayerVars:
    def __init__(self, rng, filter_shape, meansquare = False):
        self.W = mp.RawArray(ctypes.c_double, reduce(mul, filter_shape))

        w = np.frombuffer(self.W.get_obj()).reshape(filter_shape)
        np.copyto(w, np.asarray(rng.normal(size=filter_shape)))
        
        self.b = mp.RawArray(ctypes.c_double, filter_shape([0])
        
        b = np.frombuffer(self.b.get_obj())
        b[:] = 0.01

        self.mean_square = meansquare 
        if mean_square:
            self.W_ms  = mp.RawArray(ctypes.c_double, reduce(mul, filter_shape))
            w_ms = np.frombuffer(self.W.get_obj())
            w_ms[:] = 1

            self.b_ms = mp.RawArray(ctypes.c_double, filter_shape[0])
            b_ms = np.frombuffer(self.b_ms.get_obj())
            b_ms[:] = 1
        

class FullyConectedLayerVars:
    def __init__(self, rng, nb_inputs, nb_outputs, meansquare=False):
        self.shape = [nb_inputs, nb_outputs]
        self.W     = mp.RawArray(ctypes.c_double, nb_inputs * nb_outputs)

        w = np.frombuffer(self.W.get_obj()).reshape(self.shape)
        np.copyto(w, np.asarray(rng.normal(size=self.shape)))

        self.b    = mp.RawArray(ctypes.c_double, nb_outputs)

        b = np.frombuffer(self.b.get_obj())
        b[:] = 0.01

        self.mean_square = meansquare

        if meansquare:
            self.W_ms  = mp.RawArray(ctypes.c_double, nb_inputs * nb_outputs)
            w_ms = np.frombuffer(self.W.get_obj())
            w_ms[:] = 1

            self.b_ms = mp.RawArray(ctypes.c_double, nb_outputs)
            b_ms = np.frombuffer(self.b_ms.get_obj())
            b_ms[:] = 1

class FullyConectedLayer:
    def __init__(self, inputs,  fcl, activation, name_prefix):
        self.inputs = inputs

        self.W = shared(
                        np.frombuffer(fcl.W.get_obj()).reshape(fcl.shape),
                        borrow = True,
                        name = name_prefix + "_w"
                    )
        self.b = shared(
                        np.frombuffer(fcl.b.get_obj()),
                        borrow = True,
                        name = name_prefix + "_b"
                    )
        
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]

        self.meansquare_param = []

        if fcl.mean_square:
            self.W_ms = shared(
                            np.frombuffer(fcl.W_ms.get_obj()).reshape(fcl.shape),
                            borrow = True,
                            name = name_prefix+"_WMS"
                        )
            self.b_ms = shared(
                            npo.frombuffer(fcl.b_ms.get_obj()),
                            borrow = True, 
                            name = name_prefix+"_WMS"
                        )
            self.meansquare_param = [self.W_ms, self.b_ms]

class DeepQNet:
    def __init__(self, n_sorties, prefix):
        rng = np.random.RandomState(42)
        self.conv1         = ConvLayerVars(rng, 
                                constants.conv1_shape
                            )

        self.conv1_critic  = ConvLayerVars(rng, 
                                constants.conv1_shape
                            )

        self.conv2         = ConvLayerVars(rng, 
                                constants.conv2_shape
                            )

        self.conv2_critic  = ConvLayerVars(rng, 
                                constants.conv2_shape
                            )

    
        self.fcl1         = FullyConectedLayerVars(rng, 
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                True
                            )

        self.fcl1_critic = FullyConectedLayerVars(rng,
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit
                            )

        self.fcl2        = FullyConectedLayerVars(rng,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                True
                            )

        self.fcl2_critic = FullyConectedLayerVars(rng,
                                constants.fcl1_nbUnit,
                                n_sorties
                            )

        
class AgentComputation:
    def __init__(self, network, prefix):
        self.network = network

        self.inputs  = T.dtensor4(name=prefix+"_input")

        conv1        = ConvLayer(network.conv1, 
                        self.inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                    )
        conv1_critic = ConvLayer(network.conv1_critic, 
                        self.inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                    )

        conv2        = ConvLayer(network.conv2, 
                        conv1.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                    )

        conv2_critic = ConvLayer(network.conv2_critic, 
                        conv1_critic.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                    )

    
        fcl1         = FullyConectedLayer(
                                conv2.out.flatten(ndim=2),
                                network.fcl1,
                                T.nnet.relu,
                            )

        fcl1_critic  = FullyConectedLayer(
                                conv2_critic.out.flatten(ndim=2),
                                network.fcl1_critic,
                                T.nnet.relu,
                            )

        fcl2         = FullyConectedLayer(
                                fcl1.output,
                                network.fcl2,
                                None,
                            )

        fcl2_critic  = FullyConectedLayer(
                                fcl1_critic.output,
                                network.fcl2_critic,
                                None,
                            )

        self.update_critic = function([],[], updates=[
                    (self.conv1_critic.W, self.conv1.W),
                    (self.conv1_critic.b, self.conv1.b),
                    (self.conv2_critic.W, self.conv2.W),
                    (self.conv2_critic.b, self.conv2.b),
                    (self.fcl1_critic.W,  self.fcl1.W),
                    (self.fcl1_critic.b,  self.fcl1.b),
                    (self.fcl2_critic.W,  self.fcl2.W),
                    (self.fcl2_critic.b,  self.fcl2.b)])

        self.updatable = [conv1, conv2, fcl1, fcl2] 

        self.params = []
        self.meansquare_params = []
        for layer in self.updatable:
            self.params            += layer.params
            self.meansquare_params += layer.meansquare_param

        self.best_actions   = T.argmax(fcl2.output)
        self.critic_score   = T.max(fcl2_critic.output)

        self.actions = T.ivector(prefix+'_actionsVector');
        self.labels  = T.dvector(prefix+'_labels')

        self.actions_scores = fcl2.output[T.arange(self.actions.shape[0]), self.actions]
        self.error = T.mean(.5 * (self.actions_scores - self.labels)**2)

        self.gradients  = [T.grad(self.error, param)   for param in self.params] 
        #The shared "G's" from RMSProp

        self.getBestAction  = function([self.inputs],[self.best_actions])
        self.getCriticScore = function([self.inputs],[self.critic_score])

        self.learning_rate = T.dscalar()
        self.gradientsAcc = [shared(np.zeros(param.get_value().shape)) for param in self.params]
        self.clearGradients()
        accGradUpdates = [(accGradient, accGradient + gradient) \
                            for accGradient, gradient in zip(self.gradientsAcc, self.gradients)]

        self.cumulateGradient = function([self.inputs, self.actions, self.labels],
                                    [],
                                    updates = accGradUpdates)

        meansquare_update = [(ms, constants.decay_factor * ms + (1-constants.decay_factor) * T.sqr(grad))\
                                for ms, grad in zip(self.meansquare_params, self.gradientsAcc)]

        param_update  =  [(param, 
                          param - self.learning_rate * gradient / T.sqrt(square_mean + constants.epsilon_cancel)) \
                        for param, gradient, square_mean in zip(self.params, 
                                    self.gradientsAcc,
                                    self.meansquare_param
                                )]

        self.applyGradient = function([self.learning_rate],[],updates = meansquare_update + param_update)


    def clearGradients(self):
        for var in self.gradientsAcc:
            var.set_value(np.zeros(var.get_value().shape))
