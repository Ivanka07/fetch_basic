import gym
import numpy
import csv
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import datetime
import pkg_resources
import cfg_load
import warnings


'''
<body name="goal:g1" pos="0 0.0 0" >
			<site name="target0:id1" pos="0.0 0 0" size="0.02 0.0 0.02" rgba="1 0 0 1" type="sphere"></site>
</body>
'''
path = 'config.yaml' 
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
config = cfg_load.load(filepath)
world = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['file'])


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


def rotate_translate_goals(goals):
	pass


def calc_center_of_mass(goals):
	center_of_mass = [0, 0, 0]
	for g in goals:
		center_of_mass[0] += g[0] 
		center_of_mass[1] += g[1]
		center_of_mass[2] += g[2]

	center_of_mass[0] = center_of_mass[0] / len(goals)
	center_of_mass[1] = center_of_mass[1] / len(goals)
	center_of_mass[2] = center_of_mass[2] / len(goals)

	print('calculated center of  center ',)
	return center_of_mass




def add_goals_to_env_xml(file, goals, center_of_mass=[]):
	if not len(goals):
		warnings.warn('List of goals is empty')
	tree = ET.parse(file)
	root = tree.getroot()
	print('Root', root.tag)
	world = root.find('worldbody')

	if len(center_of_mass):
		center_of_mass_el = world.find('body')
		print(center_of_mass_el.attrib)
		center_of_mass_el.set('pos', '{} {} {}'.format(center_of_mass[0], center_of_mass[1], center_of_mass[2]))

	for i in range(len(goals)):
		if i > 800:
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


	
	tree.write('output.xml')



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


def write_xml(file):


	pass

goals = read_data_csv('/home/ivanna/git/generative-learning-by-demonstration/data/triangle_2018-08-23-18-00-42.bag_hand_positions.csv')
center_of_mass = calc_center_of_mass(goals)
add_goals_to_env_xml(world, goals, center_of_mass)