from rotate_suite import suite
from dm_control import viewer
import numpy as np

env = suite.load('rotate', 'easy')
action_spec = env.action_spec()


def random_policy(time_step):
    del time_step
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)


viewer.launch(env, policy=random_policy)
