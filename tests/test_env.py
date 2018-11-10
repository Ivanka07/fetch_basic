import gym
import gym_fetch_base_motions

print('Starting gym with version {}'.format(gym.__version__))
env = gym.make('FetchBase-v0')

while True:
	env.render()
