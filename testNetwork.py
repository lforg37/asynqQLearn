import network
import theano as th
from theano import tensor as T
import numpy as np

def main():
    dqn    = network.DeepQNet(1, "net",    True)
    critic = network.DeepQNet(1, "critic", False)
    computation = network.AgentComputation(dqn, critic, "")

    _actionValue = th.function(dqn.params + [computation.inputs], dqn.fcl2.output)
    def actionValue(image):
        return _actionValue(*dqn.weight_parameters, image)

    image = np.random.RandomState(32).normal(size = [4,1,84,84]).astype(np.float32)

    print(image)

    for i in range(50):
        print(actionValue(image))
        for k in range(5):
            computation.cumulateGradient(image, 0, 12, 0)

        computation.applyGradient(10**-3)

if __name__ == '__main__':
    main()
