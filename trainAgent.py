import sys
from ale_python_interface import ALEInterface
from parameters           import constants
from network import DeepQNet
from Agent import AgentProcess
from mathtools import logUniform
from RWLock import RWLock
import multiprocessing as mp
import ctypes


def main():
    if len(sys.argv) < 2:
        print("Missing rom name !")
        return
    
    romname = sys.argv[1].encode('ascii')
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

    learning_rate = 10**-3
    
    barrier = mp.Barrier(constants.nb_agent) 

    for i in range(0, constants.nb_agent):
        agentpool.append(mp.Process(target = AgentProcess, args=[rwlock, dqn, dqn_critic, T, TLock, romname, i, learning_rate, barrier]))
    
    for t in agentpool:
        t.start()

    for t in agentpool:
        t.join()

    dqn.save('network')

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
