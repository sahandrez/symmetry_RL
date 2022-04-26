"""Rotate domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from rotate_suite.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards, transformations
import numpy as np


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
OBJECT_TYPES = ['box', 'capsule', 'ellipsoid', 'cylinder']


# TODO: Add more randomization to the environment


def get_model_and_assets(object_type):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(f'assets/{object_type}.xml'), common.ASSETS


@SUITE.add('rotate')
def easy(time_limit=_DEFAULT_TIME_LIMIT, object_type='box', random=None,
         environment_kwargs=None):
    """Returns rotate easy task ."""
    assert object_type in OBJECT_TYPES, f"Available object types are {OBJECT_TYPES}"
    physics = Physics.from_xml_string(*get_model_and_assets(object_type))
    task = Rotate(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Rotate domain."""

    def orientation(self):
        """Returns the rotation matrix of the base."""
        return self.named.data.xmat['base']


class Rotate(base.Task):
    """A Rotate `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Rotate`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to a random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
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
        # TODO
        obs = collections.OrderedDict()
        obs['orientation'] = physics.orientation()
        return obs

    def get_reward(self, physics):
        # TODO
        return 0.
        # return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))