import gym
import gym_fetch_base_motions
import numpy as np

print('Starting gym with version {}'.format(gym.__version__))
env = gym.make('FetchBase-v0')

action = np.array([0.02, 0.0, 0.05, 0.0])
print('Action space ', action.shape)

while True:
	env.render()
	#observations, reward, done, info = env.step(action)
	#print('Reward ', reward)
