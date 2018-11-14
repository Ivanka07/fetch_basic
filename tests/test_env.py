import gym
import gym_fetch_base_motions
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

file = '/home/ivanna/git/gym-fetch/gym_fetch_base_motions/assets/fetch_base/output.xml'
goals = []

def get_goals_from_xml(file):
	tree = ET.parse(file)
	root = tree.getroot()
	print('Root', root.tag)
	world = root.find('worldbody')

	for body in world.findall('body'):
		name = body.get('name')
		if 'goal' in name:
			pos = body.get('pos')
			_pos = [float(s) for s in pos.split()]
			goals.append(_pos)


print('Starting gym with version {}'.format(gym.__version__))
env = gym.make('FetchBase-v0')
obs = env.reset()
gripper_position = obs[0:3]

print('Initial gripper pos = ', gripper_position)
get_goals_from_xml(file)

i = 0
reward = 0.0
while True:
	env.render()
	#calc action
	vec = goals[i] - gripper_position
	action = [vec[0], vec[1], vec[2], 2.5]
	obs, r, done, info = env.step(action)
	gripper_position = obs[0:3]
	
	i+=1
	
	print('r =  ', r, 'reward = ', reward )




