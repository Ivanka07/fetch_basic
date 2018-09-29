#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.

Each episode is selling a single banana.
"""

# core modules
import math
import pkg_resources
import random

# 3rd party modules
from gym import spaces
from gym import utils
from gym.envs.robotics import fetch_env
import cfg_load
import gym
import numpy as np


path = 'config.yaml'  
filepath = pkg_resources.resource_filename('gym_fetch_base_motions', path)
print(filepath)
config = cfg_load.load(filepath)
world = filepath = pkg_resources.resource_filename('gym_fetch_base_motions', config['ASSETS']['file'])
print(world)

class FetchBaseEnv(fetch_env.FetchEnv, utils.EzPickle):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, reward_type='sparse'):
        self.__version__ = "0.1.0"
        print("FetchBaseEnv - Version {}".format(self.__version__))
        # General variables defining the environmen

        self.observations = spaces.Box(-1., 1., shape=(13,), dtype='float32')
        self.actions = fetch_env.FetchEnv.action_space

        initial_qpos = {
            'robot0:slide0': 0.1,
            'robot0:slide1': 0.1,
            'robot0:slide2': 0.0
        }
        
        fetch_env.FetchEnv.__init__(
            self, world, has_object=False, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        print('Actions %s' % self.action_space) 
        

    def _get_reward(self):
        """Reward is given for a sold banana."""
        if self.is_banana_sold:
            return self.price - 1
        else:
            return 0.0

    def reset(self):
        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        return ob

