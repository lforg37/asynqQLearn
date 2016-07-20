import theano
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano import shared

import numpy as np

from parameters import constants
class ConvLayer:
    def __init__(self, convl, inputs, filter_shape, stride):
        self.inputs = inputs
        self.W = convl.W        
        self.b = convl.b

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
    def __init__(self, rng, filter_shape, name_prefix):
        self.W = theano.shared(
                    np.asarray(
                        rng.normal(size=filter_shape),
                    ),
                    borrow = True,
                    name   = name_prefix+"_W"
                )
        
        self.b = theano.shared(np.zeros(filter_shape[0]) + 0.01,
                                borrow = True,
                                name   = name_prefix+"_b"
                            )
        self.params = [self.W, self.b]

class FullyConectedLayerVars:
    def __init__(self, rng, nb_inputs, nb_outputs, name_prefix):
        self.W = theano.shared(
                        np.asarray(
                            rng.normal(size = [nb_inputs, nb_outputs])
                        ),
                        borrow = True,
                        name   = name_prefix + "_W"
                    )

        self.b = theano.shared(
                    np.zeros(nb_outputs) + 0.01,
                    borrow = True,
                    name = name_prefix + "_b"
                )
        self.params = [self.W, self.b]

class FullyConectedLayer:
    def __init__(self, inputs,  fcl, activation):
        self.inputs = inputs

        self.W = fcl.W
        self.b = fcl.b
        
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]


class DeepQNet:
    def __init__(self, n_sorties, prefix):
        rng = np.random.RandomState(42)
        self.conv1         = ConvLayerVars(rng, 
                        constants.conv1_shape,
                        prefix+"_conv1"
                    )
        self.conv1_critic  = ConvLayerVars(rng, 
                        constants.conv1_shape,
                        prefix+"_conv1_critic"
                    )

        self.conv2         = ConvLayerVars(rng, 
                        constants.conv2_shape,
                        prefix+"_conv2"
                    )

        self.conv2_critic  = ConvLayerVars(rng, 
                        constants.conv2_shape,
                        prefix+"_conv2_critic"
                    )

    
        self.fcl1         = FullyConectedLayerVars(rng, 
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                prefix+"_fcl1"
                            )

        self.fcl1_critic = FullyConectedLayerVars(rng,
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                prefix+"_fcl1_critic"
                            )

        self.fcl2        = FullyConectedLayerVars(rng,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                prefix+"_fcl2"
                            )

        self.fcl2_critic = FullyConectedLayerVars(rng,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                prefix+"_fcl2_critic"
                            )
        
        self.params  = self.conv1.params + \
                      self.conv2.params + \
                      self.fcl1.params  + \
                      self.fcl2.params
        self.meanSquareGrad = [shared(np.ones(param.get_value().shape)) for param in self.params]

        self.update_critic = function([],[], updates=[
                    (self.conv1_critic.W, self.conv1.W),
                    (self.conv1_critic.b, self.conv1.b),
                    (self.conv2_critic.W, self.conv2.W),
                    (self.conv2_critic.b, self.conv2.b),
                    (self.fcl1_critic.W,  self.fcl1.W),
                    (self.fcl1_critic.b,  self.fcl1.b),
                    (self.fcl2_critic.W,  self.fcl2.W),
                    (self.fcl2_critic.b,  self.fcl2.b)])

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
        
        self.params  = conv1.params + \
                      conv2.params + \
                      fcl1.params  + \
                      fcl2.params

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
        self.gradientsAcc = [shared(np.zeros(param.get_value().shape)) for param in self.network.params]
        self.clearGradients()
        accGradUpdates = [(accGradient, accGradient + gradient) \
                            for accGradient, gradient in zip(self.gradientsAcc, self.gradients)]

        self.cumulateGradient = function([self.inputs, self.actions, self.labels],
                                    [],
                                    updates = accGradUpdates)

        meansquare_update = [(ms, constants.decay_factor * ms + (1-constants.decay_factor) * T.sqr(grad))\
                                for ms, grad in zip(self.network.meanSquareGrad, self.gradientsAcc)]

        param_update  =  [(param, 
                          param - self.learning_rate * gradient / T.sqrt(square_mean + constants.epsilon_cancel)) \
                        for param, gradient, square_mean in zip(self.params, 
                                    self.gradientsAcc,
                                    self.network.meanSquareGrad
                                )]

        self.applyGradient = function([self.learning_rate],[],updates = meansquare_update + param_update)


    def clearGradients(self):
        for var in self.gradientsAcc:
            var.set_value(np.zeros(var.get_value().shape))
