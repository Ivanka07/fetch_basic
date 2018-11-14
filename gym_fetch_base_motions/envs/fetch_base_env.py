#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetch Environment for basic motions
"""

# core modules
import math
import random
import pkg_resources
import cfg_load

#config 
#from fetch_utils import world, config

# 3rd party modules
from gym import spaces
from gym.envs.robotics import robot_env
from gym.envs.robotics import utils
import gym

import numpy as np
import csv

path = 'config.yaml'  
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
config = cfg_load.load(filepath)
world = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['file'])



def distance_goal(goal_a, goal_b):
    if 'numpy.ndarray' != type(goal_a):
        goal_a = np.array(goal_a)

    if 'numpy.ndarray' != type(goal_b):
        goal_b = np.array(goal_b)

    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class Goal():
    def __init__(self, position, id, reached=False):
        self.position = position
        self.id = id
        self.reached = reached

    def print(self):
        print('Position = {}, id = {}, reached = {} '.format(self.position, self.id, self.reached))



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
            obj_range=1.0, n_substeps=20, goals_num=15
            ):

        self.__version__ = "0.1.0"
        print("FetchBaseEnv - Version {}".format(self.__version__))
        # General variables defining the environmen
        obs_shape = (goals_num * 3) + 8 
        observations_space = spaces.Box(-1., 1., shape=(obs_shape,), dtype='float32')
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
        self.num_goals=goals_num
        self.goals_state = []
        
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

        #gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        if self.multiple_goals:
            goals = self._sample_goals()
        else:
            goal = self._sample_goal() 

        obs = np.concatenate([grip_pos, grip_velp, gripper_vel])
        for g in goals:
            obs = np.concatenate([obs,g])
        print('Calculated observation %s' % obs)
        print('Shape %s' % obs.shape)
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
        
        return goals_pos

    
    def _compute_reward(self):
        """Reward is given for a perforing basic motion trajectory ."""
        """Gegeben eine liste mit den Goals, für jeden angefahrenes goal 
            wird reward addiert.
            An der stelle brauche ich weitere Ideen für Goal
        """
        print('********************* Computing reward *********************')
        distance_threshold = config['PARAMETERS']['dist_threshhold']
        print('All sites ', self.model.body_names)
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        print('Gripper position = ', grip_pos)

        print('Goal position = ', self.model.body_pos[1])
        dist = distance_goal(self.model.body_pos[1], grip_pos)
        print('Distance to the goal with pos={} is {}'.format(self.model.body_pos[1], dist))
        for gs in self.goals_state:

            dist = distance_goal(grip_pos, gs.position)
        #    print('Distance to the goal with id={} is {}'.format(gs.id, dist))
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
        print('site_pos', self.sim.data.get_site_xpos('robot0:grip'))
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        
        print('Set initial position of the endefector = ', gripper_target)
        gripper_rotation = initial_rot

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        
        for i in range(1, len(self.goal)):
            goal_obj = Goal(self.goal[i-1], 'goal:g' + str(i))
            self.goals_state.append(goal_obj)
            self.model.body_pos[i+1] = self.goal[i-1]
       
        for _ in range(10):
            self.sim.step()

        self._compute_reward()

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



    def translate_rotate_goals(self, gripper_target):

        body_names = self.model.body_names
        mass = self.model.body_pos[1]

        print('body_names', self.model.body_names)
        print('initial mass center ', mass)
        print('Gripper position ', gripper_target)

        print('----------------------------------------')

        for pos in range(2,17):
            print('Goal position before rotating ', self.model.body_pos[pos])
            self.model.body_pos[pos] = self.goal[pos-2]
            print('Goal position setting ', self.model.body_pos[pos])


        assert len(self.goal) > 2, 'Expecting more then 2 goals for basic motions'

        theta = np.radians(90)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0, s, c, 0, 0, 0, 1))
        print(R)

        result_vector = []
        center_of_mass = [0, 0, 0]
        for g in self.goal:
            center_of_mass[0] += g[0] 
            center_of_mass[1] += g[1]
            center_of_mass[2] += g[2]

        center_of_mass[0] = center_of_mass[0] / len(self.goal)
        center_of_mass[1] = center_of_mass[1] / len(self.goal)
        center_of_mass[2] = center_of_mass[2] / len(self.goal)

        #self.model.body_pos[1] = center_of_mass
        transl_vec = gripper_target - center_of_mass
        print('Translation vector', transl_vec)

        center_of_mass[0] = center_of_mass[0] + transl_vec[0]
        center_of_mass[1] = center_of_mass[1] + transl_vec[1]
        center_of_mass[2] = center_of_mass[2] + transl_vec[2]

        print('Center of mass ', center_of_mass)

        #self.model.body_pos[1] = center_of_mass

        for i in range(0, len(self.goal)):
            g = self.goal[i]


            a = np.array(g)
            a = R* a.reshape(3,1)
            a = a.reshape(1,3)
            t = [0,0,0]

            t[0] = a.item(0) 
            t[1] = a.item(1) # + config['PARAMETERS']['reachable']['offset_y']
            t[2] = a.item(2)

            g[0] = t[0] + transl_vec[0] +  config['PARAMETERS']['reachable']['offset_x']
            g[1] = t[1] + transl_vec[1]
            g[2] = t[2] + transl_vec[2]

            self.model.body_pos[i+2] = g    
            result_vector.append(t)

        self.goal = result_vector
        return center_of_mass



    def print_goals_state(self):
        for gs in self.goals_state:
            gs.print()

