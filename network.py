import theano
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano import shared

import numpy as np

from parameters import constants

class ConvLayer:
    def __init__(self, rng, inputs, filter_shape, stride, name_prefix):
        self.inputs = inputs
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
        conv_out = conv2d(input = inputs,
                    filters      = self.W,
                    filter_shape = filter_shape,
                    subsample    = stride
                ) 

        addition = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.out = T.nnet.relu(addition)
        self.params = [self.W, self.b]

class FullyConectedLayer:
    def __init__(self, rng, inputs, nb_inputs, nb_outputs, activation, name_prefix):
        self.inputs = inputs
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
        lin_output = T.dot(inputs, self.W) + self.b
        
        self.output = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]
         


class DeepQNet:
    def __init__(self, n_sorties, prefix):
        rng = np.random.RandomState(42)
        self.inputs   = T.dtensor4(name=prefix+"_input")
        conv1         = ConvLayer(rng, 
                        self.inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                        prefix+"_conv1"
                    )
        conv1_critic  = ConvLayer(rng, 
                        self.inputs,
                        constants.conv1_shape,
                        constants.conv1_strides,
                        prefix+"_conv1_critic"
                    )

        conv2         = ConvLayer(rng, 
                        conv1.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                        prefix+"_conv2"
                    )

        conv2_critic  = ConvLayer(rng, 
                        conv1_critic.out,
                        constants.conv2_shape,
                        constants.conv2_strides,
                        prefix+"_conv2_critic"
                    )

    
        fcl1         = FullyConectedLayer(rng, 
                                conv2.out.flatten(ndim=2),
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                T.nnet.relu,
                                prefix+"_fcl1"
                            )

        fcl1_critic = FullyConectedLayer(rng,
                                conv2_critic.out.flatten(ndim=2),
                                constants.cnn_output_size,
                                constants.fcl1_nbUnit,
                                T.nnet.relu,
                                prefix+"_fcl1_critic"
                            )

        fcl2        = FullyConectedLayer(rng,
                                fcl1.output,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                None,
                                prefix+"_fcl2"
                            )

        fcl2_critic = FullyConectedLayer(rng,
                                fcl1_critic.output,
                                constants.fcl1_nbUnit,
                                n_sorties,
                                None,
                                prefix+"_fcl2_critic"
                            )
        
        self.params = conv1.params + \
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
        self.meanSquareGrad = [shared(np.ones(param.get_value().shape)) for param in self.params]

        self.getBestAction  = function([self.inputs],[self.best_actions])
        self.getCriticScore = function([self.inputs],[self.critic_score])

        self.update_critic = function([],[], updates=[
                    (conv1_critic.W, conv1.W),
                    (conv1_critic.b, conv1.b),
                    (conv2_critic.W, conv2.W),
                    (conv2_critic.b, conv2.b),
                    (fcl1_critic.W,  fcl1.W),
                    (fcl1_critic.b,  fcl1.b),
                    (fcl2_critic.W,  fcl2.W),
                    (fcl2_critic.b,  fcl2.b)])

class AgentComputation:
    def __init__(self, network):
        self.network = network
        self.learning_rate = T.dscalar()
        self.gradients = [shared(np.zeros(param.get_value().shape)) for param in self.network.params]
        self.clearGradients()
        accGradUpdates = [(accGradient, accGradient + gradient) \
                            for accGradient, gradient in zip(self.gradients, self.network.gradients)]

        self.cumulateGradient = function([self.network.inputs, self.network.actions, self.network.labels],
                                    [],
                                    updates = accGradUpdates)

        meansquare_update = [(ms, constants.decay_factor * ms + (1-constants.decay_factor) * T.sqr(grad))\
                                for ms, grad in zip(self.network.meanSquareGrad, self.gradients)]

        param_update  =  [(param, 
                          param - self.learning_rate * gradient / T.sqrt(square_mean + constants.epsilon_cancel)) \
                        for param, gradient, square_mean in zip(self.network.params, 
                                    self.gradients,
                                    self.network.meanSquareGrad
                                )]

        self.applyGradient = function([self.learning_rate],[],updates = meansquare_update + param_update)


    def clearGradients(self):
        for var in self.gradients:
            var.set_value(np.zeros(var.get_value().shape))
