import network
import theano as th
from theano import tensor as T
import numpy as np
import multiprocessing as mp

def overfit(dqn, critic, image, label, ident, barrier):
    computation = network.AgentComputation(dqn, critic, "")
    _actionValue = th.function(dqn.params + [computation.inputs], dqn.fcl2.output)

    def actionValue(image):
        return _actionValue(*dqn.weight_parameters, image)

    barrier.wait()
    for i in range(100):
        print("["+str(ident)+"] Label : "+str(label)+", Value : "+str(actionValue(image)))

        for k in range(5):
            computation.cumulateGradient(image, 0, label, ident)
        computation.applyGradient(10**-3)

def main():
    dqn    = network.DeepQNet(1, "net",    True)
    critic = network.DeepQNet(1, "critic", False)
    computation = network.AgentComputation(dqn, critic, "")

    _actionValue = th.function(dqn.params + [computation.inputs], dqn.fcl2.output)
    def actionValue(image):
        return _actionValue(*dqn.weight_parameters, image)

    images = []
    labels = []
    proc = []
    nb_proc = 2
    barrier = mp.Barrier(nb_proc)
    rng = np.random.RandomState(42)
    for i in range(nb_proc) :
        image = rng.normal(size=[4,1,84,84]).astype(np.float32)
        images.append(image)
        label = 5 * i
        labels.append(label)
        p = mp.Process(target = overfit, args=[dqn, critic, image, label, i, barrier])
        proc.append(p)

    for p in proc:
        p.start()
        #p.join() #Commenter pour exécution parallèle

    for p in proc:
        p.join()

    for image, label in zip(images, labels):
        print("Expected : ", str(label), " got {:3.2f}".format(actionValue(image)[0]))

if __name__ == '__main__':
    main()