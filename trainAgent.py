import sys
from ale_python_interface import ALEInterface
from parameters           import constants
from network import DeepQNet
from Agent import AgentProcess
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

    dqn = DeepQNet(nb_actions, "mainDQN")
    
    rwlock = RWLock()
    
    agentpool = []
    
    T = mp.RawValue(ctypes.c_uint)
    T = 0
    TLock = mp.Lock()

    for i in range(0, constants.nb_agent):
        agentpool.append(mp.Process(target = AgentProcess, args=[rwlock, dqn, T, TLock, romname, i]))
    
    for t in agentpool:
        t.start()

    for t in agentpool:
        t.join()

    dqn.save('network')

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
