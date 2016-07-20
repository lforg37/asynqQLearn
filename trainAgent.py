import sys
from ale_python_interface import ALEInterface
from parameters           import shared, constants
from network import DeepQNet
from threading import Lock
from Agent import AgentThread

def main():
    print("Est-ce que ce print s'affiche ?????")
    if len(sys.argv) < 2:
        print("Missing rom name !")
        return
    
    romname = sys.argv[1].encode('ascii')
    ale = ALEInterface()
    ale.loadROM(romname)
    nb_actions = len(ale.getMinimalActionSet()) 

    dqn = DeepQNet(nb_actions, "mainDQN")
    
    agentpool = []
    lock = Lock()
    
    for i in range(0, constants.nb_thread):
        agentpool.append(AgentThread(lock, dqn, romname, i))
    
    for t in agentpool:
        t.start()

print("Est-ce que ce print s'affiche ?????")
if __name__ == "__main__":
    main()
