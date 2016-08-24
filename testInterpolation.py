from ale_python_interface import ALEInterface
from improc import BilinearInterpolator2D
from random import randrange, choice
import numpy as np

#Set up game environment
ale = ALEInterface()
ale.setInt(b'random_seed', randrange(0,256,1))

#ale.setBool(b'color_averaging', True)
ale.loadROM(b"roms/breakout.bin")
actions = ale.getMinimalActionSet()

interpolator = BilinearInterpolator2D([210,160],[84,84])
current_frame = np.empty([210, 160, 1], dtype=np.uint8)
next_state    = np.empty([84, 84, 1], dtype=np.float32)

action_repeats=20
i=0
reward=0
while i < action_repeats and not ale.game_over():
	action = choice(actions)
	reward += ale.act(action)
	ale.getScreenGrayscale(current_frame)
	interpolator.interpolate(current_frame, next_state)
	i += 1

	test_state = next_state.transpose(2,0,1)		
	import matplotlib.pyplot as plt 
	plt.subplot(1,2,1)
	plt.imshow(current_frame[:,:,0], interpolation='none', cmap='gray')
	plt.subplot(1,2,2)
	plt.imshow(test_state[0,:,:],interpolation='none', cmap='gray')
	
	plt.show() 
