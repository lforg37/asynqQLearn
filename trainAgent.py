import sys
from ale_python_interface import ALEInterface
from parameters           import constants
from network import DeepQNet
from Agent import AgentProcess
from mathtools import logUniform
from RWLock import RWLock
import multiprocessing as mp
import ctypes


def main(argv):
    if len(argv) < 2:
        print("Missing rom name !")
        return
    print("Tout est ok")    
    romname = argv[1].encode('ascii')
    ale = ALEInterface()
    ale.loadROM(romname)
    nb_actions = len(ale.getMinimalActionSet()) 

    dqn        = DeepQNet(nb_actions, "mainDQN",   True)
    dqn_critic = DeepQNet(nb_actions, "criticDQN", False)
    
    rwlock = RWLock()
    
    agentpool = []
    
    T = mp.RawValue(ctypes.c_uint)
    T.value = 0
    TLock = mp.Lock()

    learning_rate = logUniform(-4, -2)

    for i in range(0, constants.nb_agent):
        agentpool.append(mp.Process(target = AgentProcess, args=[rwlock, dqn, dqn_critic, T, TLock, romname, i, learning_rate]))
    
    for t in agentpool:
        t.start()

    for t in agentpool:
        t.join()

    dqn.save('network')

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(sys.argv)
