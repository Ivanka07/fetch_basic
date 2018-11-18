#!/usr/bin/env python

import gym
import numpy as np
import csv
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import datetime
import pkg_resources
import cfg_load
import warnings
import glob


'''
<body name="goal:g1" pos="0 0.0 0" >
			<site name="target0:id1" pos="0.0 0 0" size="0.02 0.0 0.02" rgba="1 0 0 1" type="sphere"></site>
</body>
'''


path = 'config.yaml' 
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
DATA_DIR = pkg_resources.resource_filename('gym_fetch_base_motions', 'data')
print('Filepath', DATA_DIR)
config = cfg_load.load(filepath)
world = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['file'])


def load_config():
	config={}
	config['basic'] = world
	config['limit_goals'] = config['PARAMETERS']['limit_goals']

	return config


def read_data_csv(file):
	goals = []
	with open(file, newline='') as csv_data:
		reader = csv.DictReader(csv_data, delimiter=',')
		headers = reader.fieldnames
		assert(headers[1] ==  'field.x' and headers[2] ==  'field.y' and headers[3] ==  'field.z'), 'Check header in csv file. Expected: filed.[x|y|z]'
		for line in reader:
			pos = [float(line['field.x']),float(line['field.y']),float(line['field.z'])]
			goals.append(pos)
	return goals



def rotate_translate_goals(goals, center_of_mass, angle=90):
	assert len(goals) > 2, 'Expecting more then 2 goals for basic motions'
	translated = []
	theta = np.radians(angle)
	c, s = np.cos(theta), np.sin(theta)
	R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0, s, c, 0, 0, 0, 1))
	gripper_target = [config['PARAMETERS']['gripper_pos']['_x'],
					config['PARAMETERS']['gripper_pos']['_y'], 
					config['PARAMETERS']['gripper_pos']['_z']] 
	
	result_vector = np.zeros((3,))
	transl_vec = np.array(gripper_target) - np.array(center_of_mass)
	result_vector = transl_vec + np.array(center_of_mass)

	#todo: add the x offset
	for g in goals:
		g_as_np = np.array(g)
		g_as_np = R * g_as_np.reshape(3,1)
		g_as_np = g_as_np + transl_vec.reshape(3,1) 
		g_as_np = g_as_np.reshape(1,3)
		translated.append([g_as_np[0,0], g_as_np[0,1], g_as_np[0,2]])
		
	return result_vector, translated



def calc_center_of_mass(goals):
	center_of_mass = [0, 0, 0]
	for g in goals:
		center_of_mass[0] += g[0] 
		center_of_mass[1] += g[1]
		center_of_mass[2] += g[2]

	center_of_mass[0] = center_of_mass[0] / len(goals)
	center_of_mass[1] = center_of_mass[1] / len(goals)
	center_of_mass[2] = center_of_mass[2] / len(goals)

	print('calculated center of  center ', center_of_mass)
	return center_of_mass



def add_goals_to_env_xml(input_xml, goals, output_xml, center_of_mass=[]):
	if not len(goals):
		warnings.warn('List of goals is empty')
	tree = ET.parse(input_xml)
	root = tree.getroot()
	world = root.find('worldbody')

	if len(center_of_mass):
		center_of_mass_el = world.find('body')
		center_of_mass_el.set('pos', '{} {} {}'.format(center_of_mass[0], center_of_mass[1], center_of_mass[2]))

	for i in range(len(goals)):
		if i > config['PARAMETERS']['limit_goals']:
			break

		pos = goals[i]
		body_attr ={
			'name': 'goal:g' + str(i),
			'pos': '{} {} {}'.format(pos[0], pos[1], pos[2])
		}

		site_attr ={
			'name': 'target0:id' + str(i),
			'pos': '{} {} {}'.format(0.0, 0.0, 0.0),
			'size': '{} {} {}'.format(0.02, 0.0, 0.02),
			'rgba':'{} {} {} {}'.format(1, 0, 0, 1), 
			'type': 'sphere'
		}

		body = Element('body', attrib=body_attr)
		site = SubElement(body, 'site', attrib=site_attr)
		world.append(body)

	tree.write('./output/' + output_xml + '.xml')




def downsample_goals(goals_array, num_goals=15):
	step = len(goals_array) // num_goals
	print('Step', step)
	goal_ind = 0
	updated_goal = []

	av_pose = [0,0,0]
	for i in range(len(goals_array)):

		x = goals_array[i][0]
		y = goals_array[i][1]
		z = goals_array[i][2] 

		av_pose[0] +=x 
		av_pose[1] +=y
		av_pose[2] +=z

		if i % step == 0:
			#print ('addiert pose ', av_pose)
			av_pose[0] = float(av_pose[0]) / float(step)
			av_pose[1] = float(av_pose[1]) / float(step)
			av_pose[2] = float(av_pose[2]) / float(step)
			#print ('Avarage pose ', av_pose)
			updated_goal.append(av_pose)	
			av_pose = [0,0,0]
		goal_ind+=1 
	return updated_goal



def distance_goal(goal_a, goal_b):
    if 'numpy.ndarray' != type(goal_a):
        goal_a = np.array(goal_a)

    if 'numpy.ndarray' != type(goal_b):
        goal_b = np.array(goal_b)

    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



def ids_to_pos(body_names, body_pos, goal_tag='goal:g'):
	assert len(body_names) == len(body_pos), 'Expected equal length of body_names and body_pos'
	goals_to_pos = {}
	for i in range(body_names):
		name = body_names[i]
		pos = body_pos[i]
		if goal_tag in name:
			goals_to_pos[name] = pos

	return goals_to_pos


def build_data_set(prefix=None):
	print(DATA_DIR + prefix)
	files = glob.glob(DATA_DIR + '/' + prefix + '*.csv')

	for f in files:
		print('Processing file=', f)
		output = prefix + f.split(prefix)[1].split('.csv')[0]
		goals = read_data_csv(f)
		print('Shape of goals =', len(goals) )
		center_of_mass = calc_center_of_mass(goals)
		center_of_mass, _goals = rotate_translate_goals(goals, center_of_mass)
		add_goals_to_env_xml(world, _goals, output, center_of_mass=[])


def agregate_data(data_dir, store_file):
	obs, acs, rews, ep_rets = [], [], [], [] 
	files = glob.glob(data_dir + '/' + '*.npz')
	for f in files:
		 data = np.load(f)
		 obs.append(data['obs'])
		 acs.append(data['acs'])		 
		 rews.append(data['rews'])
		 ep_rets.append(data['ep_rets'])

	obs = np.array(obs)
	acs = np.array(acs)
	rews = np.array(rews)
	ep_rets = np.array(ep_rets)
	print('obs len=%s, acts len= %s, rew Len=%s, ep_rets Len=%s' % (obs.shape, acs.shape, rews.shape, ep_rets.shape)) 
	np.savez(path, obs=obs, acs=acs, rews=rews, ep_rets=ep_rets) 



if __name__ == '__main__':
	#todo: move file into this package data
	#goals = read_data_csv('/home/ivanna/git/generative-learning-by-demonstration/data/triangle_2018-08-23-18-00-42.bag_hand_positions.csv')
	#center_of_mass = calc_center_of_mass(goals)
	#center_of_mass, _goals = rotate_translate_goals(goals, center_of_mass)
	#add_goals_to_env_xml(world, _goals, center_of_mass)
	#build_data_set(prefix='triangle')

	agregate_data('./output', './output/triangle_aggr_dataset.npz')