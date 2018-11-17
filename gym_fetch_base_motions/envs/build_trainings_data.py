import gym
import gym_fetch_base_motions
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


goals = []
print('Starting gym with version {}'.format(gym.__version__))


def init_parser():
	parser = argparse.ArgumentParser("Run models and store ")
	parser.add_argument('world',  type=str,  help='Path to the world model')
	return parser.parse_args()


def get_goals_from_xml(file):
	print('File=', file)
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


def calculate_returns(rewards, discount=0.9):
    assert rewards.shape[0] > 0
    episode_return = 0
    for i in range(0, rewards.shape[0]):
        episode_return = episode_return +  rewards[i]*discount**i
    return episode_return 


def store_as_npz(obs, acs, rews, path=''):
	obs = np.array(obs)
	acs = np.array(acs)
	rews = np.array(rews)
	ep_ret = calculate_returns(rews)

	print('obs len=%s, acts len= %s, rew Len=%s, ep_rets Len=%s' % (obs.shape, acs.shape, rews.shape, ep_ret.shape))
	file = './output/' + path
	np.savez(file, obs=obs, acs=acs, rews=rews, ep_rets=ep_ret) 

#####################################################################################################################



args = init_parser()
get_goals_from_xml(args.world)


obs, acs, rews = [], [], []
env = gym.make('FetchBase-v0')
cur_obs = env.reset()
gripper_position = np.array(cur_obs[0:3])

print('Initial gripper pos = ', gripper_position)

_path = args.world.split('triangle')[1]
npz_path = 'triangle' + _path.split('xml')[0] + 'npz'

i = 0
reward = 0.0
while True:
	env.render()
	g = np.array(goals[i]) 
	#calc action
	
	print('i=', i)
	
	vec = g.reshape(3,1) - gripper_position.reshape(3,1)
	action = [vec[0], vec[1], vec[2], 2.5]
	acs.append(action)
	cur_obs, r, done, info = env.step(action)
	obs.append(cur_obs)
	rews.append(r)
	gripper_position = np.array(cur_obs[0:3])
	i+=1
	print('r =  ', r, 'reward = ', reward )

	if i==800:
		print('Len of obs={}, acs={}, rew={}'.format(len(obs), len(acs), len(rews)))
		print('Save to=', npz_path)
		store_as_npz(obs, acs, rews, npz_path)
		break





