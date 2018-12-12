import gym
import gym_fetch_base_motions
import numpy as np
import csv
import pkg_resources
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

print('Starting gym with version {}'.format(gym.__version__))

goals = []
model_dir = pkg_resources.resource_filename('gym_fetch_base_motions', 'assets/fetch_base/')
actions_dir = pkg_resources.resource_filename('gym_fetch_base_motions', 'data/actions/')
obs_csv = pkg_resources.resource_filename('gym_fetch_base_motions', 'data/expert_traj/triangle_3_goals_obs.csv')
acs_csv = pkg_resources.resource_filename('gym_fetch_base_motions', 'data/expert_traj/triangle_3_goals_acs.csv')
rew_csv = pkg_resources.resource_filename('gym_fetch_base_motions', 'data/expert_traj/triangle_3_goals_rew.csv')



def init_parser():
	parser = argparse.ArgumentParser("Run models and store ")
	parser.add_argument('world',  type=str,  help='Path to the world model')
	parser.add_argument('actions',  type=str,  help='Path to the expert actions')
	return parser.parse_args()



def get_goals_from_xml(file):
	file = model_dir + file
	print('Current world model =', file)
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


def get_expert_actions(file):
	file = actions_dir + file
	actions = []
	with open(file, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in reader:
			action = []
			for a in row: 
				action.append(float(a))
			actions.append(action)
	return actions


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



def store_trajectory(observations, actions, rewards, separator=' '):
	assert len(observations) == len(actions) == len(rewards), 'Length of observations and action has to be same number'

	with open(obs_csv, mode='a') as obs_file:
		obs_writer = csv.writer(obs_file, delimiter=separator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for o in observations:
			obs_writer.writerow(o)

	with open(acs_csv, mode='a') as acs_file:
		acs_writer = csv.writer(acs_file, delimiter=separator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for a in actions:
			acs_writer.writerow(a)

	with open(rew_csv, mode='a') as rew_file:
		rew_writer = csv.writer(rew_file, delimiter=separator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for r in rewards:
			rew_writer.writerow([r])
		


#####################################################################################################################


args = init_parser()
get_goals_from_xml(args.world)
actions = get_expert_actions(args.actions)

print('Actions length')

obs, acs, rews = [], [], []
env = gym.make('FetchBase-v0')
cur_obs = env.reset()
gripper_position = np.array(cur_obs[0:3])
print('Initial gripper pos = ', gripper_position)

#_path = args.world.split('triangle')[1]
#npz_path = 'triangle' + _path.split('xml')[0] + 'npz'
i = 0

while True:
	env.render()

#	print('i=', i)
	goal = np.array(actions[i])
	vec = goal.reshape(3,1) - gripper_position.reshape(3,1)
	ac = [vec[0], vec[1], vec[2], 2.5]
	acs.append(ac)
	cur_obs, r, done, info = env.step(ac)
	print('Cur_obs=', len(Cur_obs))
	obs.append(cur_obs)
	rews.append(r)
	gripper_position = np.array(cur_obs[0:3])
	
	i+=1
	print('Reward = ', r)
	if i==450:
		print('******************** Reached episode end ********************')
		store_trajectory(obs, acs, rews)
		#print('Len of obs={}, acs={}, rew={}'.format(len(obs), len(acs), len(rews)))
		#print('Save to=', npz_path)
		#store_as_npz(obs, acs, rews, npz_path)
		break





