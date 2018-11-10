#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetch Environment for basic motions
"""

# core modules
import math
import pkg_resources
import random

# 3rd party modules
from gym import spaces
from gym.envs.robotics import robot_env
from gym.envs.robotics import utils
import cfg_load
import gym
import numpy as np
import csv

path = 'config.yaml'  
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
config = cfg_load.load(filepath)
world = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['file'])


class FetchBaseEnv(robot_env.RobotEnv, gym.utils.EzPickle):
    '''
    Define a Fetch Environment for Learning basic motions.

    Observation Space of the environment contains 13 elements:
        -   3D position of the enefector   
        -   3 velocities for every direction
        -   Gripper state
    '''


    def __init__(
            self, reward_type='sparse', multiple_goals=True, goal_pattern='triange',
            gripper_extra_height=0.0, distance_threshold=0.05, block_gripper=False,
            obj_range=1.0, n_substeps=20
            ):

        self.__version__ = "0.1.0"
        print("FetchBaseEnv - Version {}".format(self.__version__))
        # General variables defining the environmen
        observations_space = spaces.Box(-1., 1., shape=(10,), dtype='float32')
        actions_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')

        initial_qpos = {
            'robot0:slide0': 0.406,
            'robot0:slide1': 0.406,
            'robot0:slide2': 0.0
        }


        initial_rot =  np.array([1., 0., 0.1, 0.0])

        print('Actions %s' % actions_space) 

        self.gripper_extra_height = gripper_extra_height 
        self.block_gripper = block_gripper
        self.obj_range = obj_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.multiple_goals = multiple_goals
        self.site_prefix = 'target0'
        self.num_goals=15

        #in case of multiple_goals=True, self.goal is a list
        self.goal = self.set_goal(multiple_goals, goal_pattern)

        super(FetchBaseEnv, self).__init__(
            model_path=world, initial_qpos=initial_qpos,
            initial_rot=initial_rot,
            n_actions=4, n_substeps=n_substeps,
            action_space=actions_space, 
            observation_space=observations_space)
        
        gym.utils.EzPickle.__init__(self)


    def set_goal(self, multiple_goals=True, pattern='triangle'):
        print('----Setting goal---')
        goal = []
        #todo -> in config
        goals_array = self.get_goals_from_file('/home/ivanna/git/generative-learning-by-demonstration/data/triangle_2018-08-23-17-59-45.bag_hand_positions.csv')
        step = len(goals_array) // self.num_goals
        print('Step', step)
        goal_ind = 0
        
        av_pose = [0,0,0]
        for i in range(len(goals_array)):
            x = goals_array[i][0]
            y = goals_array[i][1]
            z = goals_array[i][2] 
            
            av_pose[0] +=x 
            av_pose[1] +=y
            av_pose[2] +=z
            print('I', i)
            if i % step == 0:
                print ('addiert pose ', av_pose)
                av_pose[0] = float(av_pose[0]) / float(step)
                av_pose[1] = float(av_pose[1]) / float(step)
                av_pose[2] = float(av_pose[2]) / float(step)
                print ('Avarage pose ', av_pose)
                goal.append(av_pose)
                av_pose = [0,0,0]
                goal_ind+=1 
        print('Goals ',  len(goal[0]))
        self.add_goals_to_env(goal)
        return goal        

    #Extenstion methods
        
    def _get_obs(self):

        # positions
        # wir brauchen eine liste mit goals
        # die goals sollen gesamplet werden

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        if self.multiple_goals:
            goals = self._sample_goals()
        else:
            goal = self._sample_goal() 

        obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel])
        print('Calculated observation %s' % obs)
        return obs.copy()

    
    def _sample_goals(self) :
        """
        We need this method for building observations
        Returns: list with goals positions
        """
        goals_pos = []

        for i in range(1, self.num_goals+1):
            _site = self.site_prefix +  ':id' + str(i)
            goal_pos = self.sim.data.get_site_xpos(_site)
            goals_pos.append(goal_pos)
        print('Goals', goals_pos)
        
        return goals_pos

    
    def _get_reward(self):
        """Reward is given for a perforing basic motion trajectory ."""
        """Gegeben eine liste mit den Goals, für jeden angefahrenes goal 
            wird reward addiert.
            An der stelle brauche ich weitere Ideen für Goal
        """
        return 0.0


    def reset(self):
        return self._get_state()

    
    def _render_callback(self):
       # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        
        for i in range(1, self.num_goals+1):
            _site = self.site_prefix +  ':id' + str(i)
            site_id = self.sim.model.site_name2id(_site)
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()


    def _get_state(self):
        """Get the observation."""
        return self._get_obs()


    def _env_setup(self, initial_qpos, initial_rot):
        '''
            Render goals here acording to the basic motion
        '''
        print('Initial env setup')

        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        print('Set initial position of the endefector = ', gripper_target)
        
        gripper_rotation = initial_rot
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        
        for _ in range(10):
            self.sim.step()
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()


    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()


    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


  # def _reset_sim(self):
  #     self.sim.set_state(self.initial_state)
  #    return True


    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        # desired goal has to be a list with goals
        #if all of goals are reached
        return False
        

    def get_goals_from_file(self, file):

        goals_array = []
        with open(file, newline='') as csv_data:
            reader = csv.DictReader(csv_data, delimiter=',')
            headers = reader.fieldnames
            assert(headers[1] ==  'field.x' and headers[2] ==  'field.y' and headers[3] ==  'field.z'), 'Check header in csv file. Expected: filed.[x|y|z]'

            for line in reader:
                pos = [float(line['field.x']),float(line['field.y']),float(line['field.z'])]
                goals_array.append(pos)
        return goals_array


    def add_goals_to_env(self, goals_pos):
        print('Rendering {} goals'.format(len(goals_pos)))
        for i in range(1, len(goals_pos)):
            print('i ', i)
            self.sim.data.set_mocap_pos('mocap:g' + str(i), goals_pos[i-1])