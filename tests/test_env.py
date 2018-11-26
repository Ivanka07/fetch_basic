import gym
import numpy as np
import math
import gym_fetch_base_motions
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import csv

print('Starting gym with version {}'.format(gym.__version__))


goals = []

env = gym.make('FetchBase-v0')
obs = env.reset()
#print('Observation=', obs)
print('Observation length', len(obs))
_g = obs[8:len(obs)]

print('Goals length', len(_g))
goals = []
for i in range(0, len(_g), 4):
	print(i, i+1, i+2)
	goal = [_g[i], _g[i+1], _g[i+2]]
	goals.append(goal)

gripper_position = obs[0:3]
print('Gripper position = ', gripper_position)
print(goals)
i = 0
cumul_rew = 0
while True:
	env.render()
	#calc action
	vec = goals[i] - gripper_position
	action = [vec[0], vec[1], vec[2], 2.5]
	obs, reward, done, info = env.step(action)
	gripper_position = obs[0:3]

	if reward > cumul_rew:
		cumul_rew = reward
		i +=1