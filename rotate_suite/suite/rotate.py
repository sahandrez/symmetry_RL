"""Rotate domain."""

import collections
import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from rotate_suite.suite import common
from dm_control.utils import containers


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
OBJECT_TYPES = ['box', 'capsule', 'ellipsoid', 'cylinder']


def get_model_and_assets(object_type):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(f'assets/{object_type}.xml'), common.ASSETS


def make_task(time_limit=_DEFAULT_TIME_LIMIT, object_type='box',
              random=None, environment_kwargs=None, render_kwargs=None):
    """Returns a rotate task ."""
    assert object_type in OBJECT_TYPES, f"Available object types are {OBJECT_TYPES}"
    physics = Physics.from_xml_string(*get_model_and_assets(object_type))
    task = Rotate(random=random, randomize_goal=False, render_kwargs=render_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('rotate')
def box_easy(**kwargs):
    assert 'object_type' not in kwargs
    return make_task(object_type='box', **kwargs)


@SUITE.add('rotate')
def capsule_easy(**kwargs):
    assert 'object_type' not in kwargs
    return make_task(object_type='capsule', **kwargs)


@SUITE.add('rotate')
def cylinder_easy(**kwargs):
    assert 'object_type' not in kwargs
    return make_task(object_type='cylinder', **kwargs)


@SUITE.add('rotate')
def ellipsoid_easy(**kwargs):
    assert 'object_type' not in kwargs
    return make_task(object_type='ellipsoid', **kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Rotate domain."""

    def orientation(self):
        """Returns the rotation matrix of the base."""
        return self.named.data.xmat['base']


class Rotate(base.Task):
    """A Rotate `Task` to swing up and balance the pole."""

    def __init__(self, random=None, randomize_goal=False,
                 observation_key='pixels', render_kwargs=None):
        """Initialize an instance of `Rotate`."""
        super().__init__(random=random)
        if render_kwargs is None:
            render_kwargs = {}
        if 'camera_id' not in render_kwargs.keys():
            render_kwargs['camera_id'] = 'lookat'

        self.randomize_goal = randomize_goal
        self.render_kwargs = render_kwargs
        self._observation_key = observation_key
        self._reward_coeff = 0.1

        self._current_obs = None
        self._goal_image = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to a random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
        if self.randomize_goal:
            # TODO: Add randomized goals for the hard task
            raise NotImplementedError
        else:
            goal = [0.0, 0.0, 0.0]

        # Set the goal image
        physics.named.data.qpos['hinge_1'] = goal[0]
        physics.named.data.qpos['hinge_2'] = goal[1]
        physics.named.data.qpos['hinge_3'] = goal[2]
        self._goal_image = self.get_observation(physics)[self._observation_key]

        # Set the initial orientation of the object
        physics.named.data.qpos['hinge_1'] = self.random.uniform(-np.pi, np.pi)
        physics.named.data.qpos['hinge_2'] = self.random.uniform(-np.pi, np.pi)
        physics.named.data.qpos['hinge_3'] = self.random.uniform(-np.pi, np.pi)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating the base orientation
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Rotate physics.

        Returns:
          A `dict` of observation.
        """
        obs = collections.OrderedDict()
        self._current_obs = physics.render(**self.render_kwargs)
        obs[self._observation_key] = self._current_obs
        return obs

    def get_reward(self, physics):
        goal_dist = np.abs(self._goal_image - self._current_obs).mean()
        reward = 1.0 - np.tanh(self._reward_coeff * goal_dist)
        return reward
