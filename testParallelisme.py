import network
import theano as th
from theano import tensor as T
import numpy as np
import multiprocessing as mp
import sys
import os

def overfit(dqn, critic, image, label, label2, ident, barrier, mutex):
    os.system('taskset -p -c ' + str(ident) + ' ' + str(os.getpid()))
    computation = network.AgentComputation(dqn, critic, "")
    _actionValue = th.function(dqn.params + [computation.inputs], dqn.fcl2.output)
    def actionValue(image):
        return _actionValue(*dqn.weight_parameters, image)

    barrier.wait()
    for i in range(10000):
        vals = actionValue(image)
        sys.stdout.write('[{0:d}] Iteration {5:d}\n\tlabel 1 : {1:3d}, value : {2:8.3f}\n\tlabel 2 : {3:3d}, value : {4:8.3f}\n'.format(ident, label, vals[0], label2, vals[1], i))
        sys.stdout.flush()

        for k in range(5):
            computation.cumulateGradient(image, 0, label, ident)
            computation.cumulateGradient(image, 1, label2, ident)

        with mutex:
            computation.applyGradient(10**-4)

def main():
    dqn    = network.DeepQNet(2, "net",    True)
    critic = network.DeepQNet(2, "critic", False)
    computation = network.AgentComputation(dqn, critic, "")

    _actionValue = th.function(dqn.params + [computation.inputs], dqn.fcl2.output)
    def actionValue(image):
        return _actionValue(*dqn.weight_parameters, image)

    images  = []
    labels  = []
    labels2 = []
    proc = []
    nb_proc = 2
    barrier = mp.Barrier(nb_proc)
    rng = np.random.RandomState(42)
    mutex = mp.Lock()
    for i in range(nb_proc) :
        image = rng.normal(size=[4,1,84,84]).astype(np.float32)
        images.append(image)
        label  = 5 * i +1
        label2 = 3 * i * i +2
        labels.append(label)
        labels2.append(label2)
        p = mp.Process(target = overfit, args=[dqn, critic, image, label, label2, i, barrier, mutex])
        proc.append(p)

    for p in proc:
        p.start()
        #p.join() #Commenter pour exécution parallèle

    for p in proc:
        p.join()

    for image, label, label2 in zip(images, labels, labels2):
        print("Expected 0 : ", str(label), " got {:3.2f}".format(actionValue(image)[0]))
        print("Expected 1 : ", str(label2), " got {:3.2f}".format(actionValue(image)[1]))

if __name__ == '__main__':
    main()
