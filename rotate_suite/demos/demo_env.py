import argparse
import numpy as np
import cv2

from rotate_suite import suite


parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='rotate')
parser.add_argument('--task', type=str, default='box_easy')
parser.add_argument('--steps', type=int, default=10000)
args = parser.parse_args()


env = suite.load(args.domain, args.task)
action_spec = env.action_spec()


def random_policy(time_step):
    del time_step
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)


time_step = env.reset()
for t in range(args.steps):
    if time_step.last():
        time_step = env.reset()
    # action = random_policy(time_step)
    action = np.array([0., 0.5, 0.])
    time_step = env.step(action)
    print(f"Reward: {time_step.reward:.2f}")
    obs = np.float32(time_step.observation['pixels'] / 255.0)
    cv2.imshow('obs', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    # cv2.imshow('goal', cv2.cvtColor(env.task._goal_image, cv2.COLOR_RGB2BGR))
    if t % 10 == 0:
        obs = time_step.observation['pixels']
        cv2.imwrite(f'images/obs_{t}.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'images/goal_{t}.png', cv2.cvtColor(env.task._goal_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
