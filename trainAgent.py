import sys
from ale_python_interface import ALEInterface
from parameters           import constants
from network import DeepQNet
from Agent import AgentThread
from RWLock import RWLock


def main():
    if len(sys.argv) < 2:
        print("Missing rom name !")
        return
    
    romname = sys.argv[1].encode('ascii')
    ale = ALEInterface()
    ale.loadROM(romname)
    nb_actions = len(ale.getMinimalActionSet()) 

    dqn = DeepQNet(nb_actions, "mainDQN")
    
    sem = BoundedSemaphore(constants.nb_agent) 
    
    agentpool = []

    for i in range(0, constants.nb_thread):
        agentpool.append(AgentThread(lock, dqn, romname, i))
    
    for t in agentpool:
        t.start()

if __name__ == "__main__":
    main()
