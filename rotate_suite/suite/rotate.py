"""Rotate domain."""

import collections
import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from rotate_suite.suite import common
from dm_control.utils import containers
import dm_control.mujoco.wrapper.mjbindings.enums as enums


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
OBJECT_TYPES = ['box', 'capsule', 'ellipsoid', 'cylinder']


def get_model_and_assets(object_type):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(f'assets/{object_type}.xml'), common.ASSETS


def make_task(time_limit=_DEFAULT_TIME_LIMIT,
              object_type='box',
              random=None,
              randomize_goal=False,
              randomize_model_shape=False,
              randomize_model_color=False,
              environment_kwargs=None,
              render_kwargs=None):
    """Returns a rotate task ."""
    assert object_type in OBJECT_TYPES, f"Available object types are {OBJECT_TYPES}"
    physics = Physics.from_xml_string(*get_model_and_assets(object_type))
    task = Rotate(random=random,
                  randomize_goal=randomize_goal,
                  randomize_model_shape=randomize_model_shape,
                  randomize_model_color=randomize_model_color,
                  render_kwargs=render_kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


def assert_task_kwargs(**kwargs):
    assert (
            'object_type' not in kwargs
            and 'randomize_goal' not in kwargs
            and 'randomize_model_shape' not in kwargs
            and 'randomize_model_color' not in kwargs
    ), "Invalid kwargs."


@SUITE.add('rotate')
def box_easy(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='box', **kwargs)


@SUITE.add('rotate')
def capsule_easy(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='capsule', **kwargs)


@SUITE.add('rotate')
def cylinder_easy(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='cylinder', **kwargs)


@SUITE.add('rotate')
def ellipsoid_easy(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='ellipsoid', **kwargs)


@SUITE.add('rotate')
def box_medium(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='box', randomize_model_shape=True, **kwargs)


@SUITE.add('rotate')
def capsule_medium(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='capsule', randomize_model_shape=True, **kwargs)


@SUITE.add('rotate')
def cylinder_medium(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='cylinder', randomize_model_shape=True, **kwargs)


@SUITE.add('rotate')
def ellipsoid_medium(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='ellipsoid', randomize_model_shape=True, **kwargs)


@SUITE.add('rotate')
def box_hard(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='box', randomize_model_color=True, **kwargs)


@SUITE.add('rotate')
def capsule_hard(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='capsule', randomize_model_color=True, **kwargs)


@SUITE.add('rotate')
def cylinder_hard(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='cylinder', randomize_model_color=True, **kwargs)


@SUITE.add('rotate')
def ellipsoid_hard(**kwargs):
    assert_task_kwargs(**kwargs)
    return make_task(object_type='ellipsoid', randomize_model_color=True, **kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Rotate domain."""

    def orientation(self):
        """Returns the rotation matrix of the base."""
        return self.named.data.xmat['base']


class Rotate(base.Task):
    """A Rotate `Task` to swing up and balance the pole."""

    def __init__(self, random=None,
                 randomize_goal=False,
                 randomize_model_shape=False,
                 randomize_model_color=False,
                 observation_key='pixels',
                 render_kwargs=None):
        """Initialize an instance of `Rotate`."""
        super().__init__(random=random)
        if render_kwargs is None:
            render_kwargs = dict(height=84, width=84)
        if 'camera_id' not in render_kwargs.keys():
            render_kwargs['camera_id'] = 'lookat'

        self.randomize_goal = randomize_goal
        self.randomize_model_shape = randomize_model_shape
        self.randomize_model_color = randomize_model_color
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
            raise NotImplementedError
        else:
            goal = [0.0, 0.0, 0.0]

        if self.randomize_model_shape:
            self.set_random_model_shape(physics)

        if self.randomize_model_color:
            self.set_random_model_color(physics)

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

    def set_random_model_shape(self, physics):
        obj_type = physics.named.model.geom_type['base']
        if obj_type in [enums.mjtGeom.mjGEOM_BOX]:
            length = np.random.uniform(0.2, 0.4)
            random_size = np.array([length, length, length])
        elif obj_type in [enums.mjtGeom.mjGEOM_CAPSULE, enums.mjtGeom.mjGEOM_CYLINDER]:
            # Size for these geom types need a dummy variable at the end
            radius = np.random.uniform(0.15, 0.4)
            length = np.random.uniform(0.2, 0.35)
            random_size = np.array([radius, length, 0.0])
        elif obj_type in [enums.mjtGeom.mjGEOM_ELLIPSOID]:
            random_size = np.random.uniform(0.2, 0.5, size=(3,))
        else:
            raise NotImplementedError
        physics.named.model.geom_size['base'] = random_size


    def set_random_model_color(self, physics):
        random_color = np.random.uniform(0., 1., size=(4,))
        random_color[-1] = 1.
        physics.named.model.geom_rgba['base'] = random_color

