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
#import fetch_utils

# 3rd party modules
from gym import spaces
from gym.envs.robotics import robot_env
from gym.envs.robotics import utils
import gym

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

import numpy as np
import csv
import sys

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


def ids_to_pos(body_names, body_pos, goal_tag='goal:g'):
    assert len(body_names) == len(body_pos), 'Expected equal length of body_names and body_pos'

    goals_to_pos = {}
    for i in range(len(body_names)):
        name = body_names[i]
        pos = body_pos[i]
        if goal_tag in name:
            goals_to_pos[name] = pos           
    return goals_to_pos

def get_goals_from_xml(world):

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


class Goal():
    def __init__(self, id, position, reached=False):
        self.id = id
        self.position = position
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
            self, reward_type='sparse', gripper_extra_height=0.0, 
            distance_threshold=0.05, block_gripper=False,
            obj_range=1.0, n_substeps=20
            ):
        self.__version__ = "0.1.0"
        print("FetchBaseEnv - Version {}".format(self.__version__))
        # General variables defining the environmen
        # obs: 3d position of goals + reached/unreached state + 
        self.obs_shape = (config['PARAMETERS']['limit_goals'] * 4) + 8 
       # print('Observation spase has to be =', self.obs_shape)
        self.act_shape = config['PARAMETERS']['action_shape']

        observations_space = spaces.Box(-1., 1., shape=(self.obs_shape,), dtype='float32')
        actions_space = spaces.Box(-1., 1., shape=(self.act_shape,), dtype='float32')

        initial_qpos = {
            'robot0:slide0': 0.406,
            'robot0:slide1': 0.406,
            'robot0:slide2': 0.0
        }

        initial_rot =  np.array([1., 0., 0.1, 0.0])

        self.gripper_extra_height = gripper_extra_height 
        self.block_gripper = block_gripper
        self.obj_range = obj_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.site_prefix = 'goal:g'
        self.num_goals = config['PARAMETERS']['limit_goals']
        self.goals = []
        self.reward = 0.0

        super(FetchBaseEnv, self).__init__(
            model_path=world, initial_qpos=initial_qpos,
            initial_rot=initial_rot,
            n_actions=4, n_substeps=n_substeps,
            action_space=actions_space, 
            observation_space=observations_space)     
        gym.utils.EzPickle.__init__(self)


    #Extenstion methods
        
    def _get_obs(self):

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        #gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        obs = np.concatenate([grip_pos, grip_velp, gripper_vel])

        for g in self.goals:
            goal = [g.position[0], g.position[1], g.position[2], g.reached]
            obs = np.concatenate([obs, goal])
#        print('Calculated observation at the end %s' % obs)
        #print('Shape %s' % obs.shape)
        return obs.copy()

    
    def _sample_goals(self):
        print('Sampling goals')
        body_names = self.model.body_names
        body_pos = self.model.body_pos
        names_to_pos = ids_to_pos(body_names, body_pos)
        self.goals = []
        for k,v in names_to_pos.items():
            random = np.random.uniform(-0.0015, 0.0015, size=3)
            v[0] = v[0]+random[0]
            v[1] = v[1]+random[1]
            v[2] = v[2]+random[2]      
            goal_obj = Goal(k, v, False)
            self.goals.append(goal_obj)
        return self.goals

    
    def _compute_reward(self):
        """Reward is given for a perforing basic motion trajectory .
           R is given for every reached goal in the trajectory
        """
       # print('********************* Computing reward ****************************')
        distance_threshold = config['PARAMETERS']['dist_threshhold']
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
       # print('Gripper position = ', grip_pos)

       # print('First goal position = ', self.goals[1])
    #   dist = distance_goal(self.model.body_pos[1], grip_pos)
    #   print('Distance to the goal with pos={} is {}'.format(self.model.body_pos[1], dist))
        
        reward = 0.0
        factor = -1.0
        only_first_unreached = True
        for i in range(len(self.goals)):
            dist = distance_goal(grip_pos, self.goals[i].position)
            print('Distance to the goal with id={} is {}'.format(self.goals[i].id, dist))
            print('Is the goal with id={} reached {}'.format(self.goals[i].id, self.goals[i].reached))               
            if not self.goals[i].reached:
                if dist < distance_threshold:
                    #reward += 1.0
                    self.goals[i].reached = True
                    reward = 0.0
                else:
                    reward += dist * factor
                only_first_unreached = False
            else:
                reward +=1           

                        
        print('Reward computed = ', reward)
        return reward


    def reset(self):
        '''
        sample new goals after
        compute new obs
        '''
        self._sample_goals()
        return self._get_obs()

    
    def _render_callback(self):
       # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
       # print('Number of goals=', self.num_goals)
        for i in range(self.num_goals):
            _site = 'target0:id'  + str(i)
        #    print('Site=', _site)
            site_id = self.sim.model.site_name2id(_site)
            self.sim.model.site_pos[site_id] = self.goals[i].position - sites_offset[i+1]
        self.sim.forward()


    #do I need this method actually?
    def _get_state(self):
        """Get the observation."""
        return self._get_obs()


    def _env_setup(self, initial_qpos, initial_rot):
        '''
            Render goals here acording to the basic motion
        '''
        print('************* Initial env setup *************')

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
        
        self._sample_goals()
        assert len(self.goals) >0 

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
        '''
        For basic motions we consider goals that build the trajectory for the figure to mimic
        '''
        #goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return self._sample_goals().copy()


    def _is_success(self, achieved_goal, desired_goal):
        '''
        Hard condition: Success is True, when all of the goals are reached
        We will try soft condition: if 75% of all goals are reached -> done
        '''
        has_to_be_reached = 0.75 * self.num_goals
        is_reached = 0
        for goal in self.goals:
            if goal.reached:
                is_reached +=1
        if has_to_be_reached <= is_reached:
            print('Consider as done! is_reached=', is_reached) 
        return has_to_be_reached <=is_reached

    def print_goals_state(self):
        for gs in self.goals:
            gs.print()
