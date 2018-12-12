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
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



'''
<body name="goal:g1" pos="0 0.0 0" >
			<site name="target0:id1" pos="0.0 0 0" size="0.02 0.0 0.02" rgba="1 0 0 1" type="sphere"></site>
</body>
'''

path = 'config.yaml' 
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
DATA_DIR = pkg_resources.resource_filename('gym_fetch_base_motions', 'data/')
MODELS_DIR = pkg_resources.resource_filename('gym_fetch_base_motions', 'assets/fetch_base/')
print('Filepath', DATA_DIR)
config = cfg_load.load(filepath)
world = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['basic'])


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



def rotate_translate_goals(goals, center_of_mass, downsampling=True, angle=90):
	assert len(goals) > 2, 'Expecting more then 2 goals for basic motions'
	_goals = []
	_actions = []
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
	i = 0
	for g in goals:
		g_as_np = np.array(g)
		g_as_np = R * g_as_np.reshape(3,1)
		g_as_np = g_as_np + transl_vec.reshape(3,1) 
		g_as_np = g_as_np.reshape(1,3)
		
		#if i % 80 == 0:
		_goals.append([g_as_np[0,0] - config['PARAMETERS']['reachable']['offset_x'], g_as_np[0,1], g_as_np[0,2]])
		_actions.append([g_as_np[0,0] - config['PARAMETERS']['reachable']['offset_x'], g_as_np[0,1], g_as_np[0,2]])
		i+=1
	
	return result_vector, _goals, _actions



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

	if 'csv' in output_xml:
		output_xml = output_xml.split('csv')[0]

	print('Input xml is=', input_xml)
	print('Goals', goals[1:2])
	tree = ET.parse(input_xml)
	root = tree.getroot()
	world = root.find('worldbody')

	#if len(center_of_mass):
	#	center_of_mass_el = world.find('body')
	#	center_of_mass_el.set('pos', '{} {} {}'.format(center_of_mass[0], center_of_mass[1], center_of_mass[2]))

	for i in range(len(goals)):
		if i > config['PARAMETERS']['limit_goals']:
			break

		pos = goals[i]
		print('Goal=', pos)
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
	print('adding new xml file')
	tree.write( output_xml + '.xml')



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

def build_triangle_from_goals(goals):
	
	min_y = []
	max_y = []
	max_z = []

	min_y_dist = 100
	max_y_dist = 0
	max_z_dist = 0

	_goals = []

	for goal in goals:
		if goal[1] < min_y_dist:
			min_y_dist = goal[1]
			min_y = goal
		
		if goal[1] > max_y_dist:
			max_y_dist = goal[1]
			max_y = goal

		if goal[2] > max_z_dist:
			max_z_dist = goal[2]
			max_z = goal
	print('Found usefull goals', max_z, min_y, max_y)

	_goals.append(min_y)
	_goals.append(max_y)
	_goals.append(max_z)

	return _goals



def build_data_set(prefix=None):
	print(DATA_DIR + prefix)
	files = glob.glob(DATA_DIR + 'triangles/' + prefix + '*.csv')

	for f in files:
		print('Processing file=', f)
		name = f.split('/')[8].split('.csv')[0]
		output =  MODELS_DIR +  name
		actions_file = DATA_DIR +  'actions/' +  name + '_actions.csv'  
		goals = read_data_csv(f)
		center_of_mass = calc_center_of_mass(goals)
		center_of_mass, _goals, actions = rotate_translate_goals(goals, center_of_mass)
		store_as_csv(actions, actions_file)
		_goals = build_triangle_from_goals(_goals)
		add_goals_to_env_xml(world, _goals, output, center_of_mass=[])


def agregate_data(data_dir, store_file):
	obs, acs, rews, ep_rets = [], [], [], [] 
	files = glob.glob(data_dir + '/' + '*.npz')
	for f in files:
		 data = np.load(f)
		 for o in data['obs']:
		 	obs.append(o)
		 #acs.append(data['acs'])		 
		 #rews.append(data['rews'])
		 #ep_rets.append(data['ep_rets'])

	obs = np.array(obs)
	acs = np.array(acs)
	rews = np.array(rews)
	ep_rets = np.array(ep_rets)
	print('obs len=%s, acts len= %s, rew Len=%s, ep_rets Len=%s' % (obs.shape, acs.shape, rews.shape, ep_rets.shape)) 
	#np.savetxt('observations.csv', obs, fmt='%s')
	
	with open('observations.csv', mode='w') as employee_file:
		employee_writer = csv.writer(employee_file, delimiter=' n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for o in obs:
			employee_writer.writerow(o)
	#np.savez(store_file, obs=obs, acs=acs, rews=rews, ep_rets=ep_rets) 


def store_as_csv(actions, file, separator=' '):
	with open(file, mode='a') as acs_file:
		acs_writer = csv.writer(acs_file, delimiter=separator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for a in actions:
			acs_writer.writerow(a)

def plot(ar):

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# Data for a three-dimensional line
	zline = np.linspace(0, 15, 1000)
	xline = np.sin(zline)
	yline = np.cos(zline)
	ax.plot3D(xline, yline, zline, 'gray')

	# Data for three-dimensional scattered points
	zdata = 15 * np.random.random(100)
	xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
	ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
	ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
	plt.show()
	

if __name__ == '__main__':
	#todo: move file into this package data
	#goals = read_data_csv('/home/ivanna/git/generative-learning-by-demonstration/data/' + cur_csv_file)
	#center_of_mass = calc_center_of_mass(goals)
	#center_of_mass, _goals = rotate_translate_goals(goals, center_of_mass)
	
	#add_goals_to_env_xml(world, _goals, cur_csv_file, center_of_mass)
	#agregate_data('./output', './output/triangle_aggr_dataset.npz')
	build_data_set(prefix='triangle')
	#plot(None)
	