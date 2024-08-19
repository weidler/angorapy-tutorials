import gym
from angorapy import make_env
from angorapy.common.const import VISION_WH, SHADOWHAND_MAX_STEPS
from angorapy.common.senses import Sensation
from angorapy.environments.hand.shadowhand import BaseShadowHandEnv, get_fingertip_distance, DEFAULT_INITIAL_QPOS
import numpy as np
from angorapy.environments.utils import robot_get_obs


def thumbsup(env, info) -> float:
    """Reward function for the thumbs up task. The reward is 1 if the thumb is raised and 0 if the thumb is lowered."""
    return 1


THUMBSUP_BASE = {
    "thumb_tip_height_threshold": 0.03
}


class DexterousThumbsUp(BaseShadowHandEnv):
    """A task where the agent has to raise or low its thumb based on a goal provided as a boolean flag."""

    def __init__(self, initial_qpos, distance_threshold, **kwargs):
        super().__init__(initial_qpos, distance_threshold, vision=False, **kwargs)

        self._thumb_tip = self.model.site("robot0:S_thtip")

    def _set_default_reward_function_and_config(self):
        self.reward_function = thumbsup
        self.reward_config = THUMBSUP_BASE

    def _sample_goal(self) -> np.array:
        """Sample a new goal and return it. The goal is a boolean flag indicating whether the thumb should be raised or
        lowered. A 1 indicates that the thumb should be raised, a 0 indicates that the thumb should be lowered."""
        return np.random.binomial(1, 0.5, size=[1])

    def step(self, action: np.ndarray):
        """Perform one step."""
        obs, reward, terminated, truncated, info = super().step(action)
        done = (terminated or truncated)

        if done:
            info['success'] = int(np.allclose(obs['achieved_goal'], obs['desired_goal']))

        return obs, reward, terminated, truncated, info

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal == desired_goal

    def assert_reward_setup(self):
        """Assert whether the reward config fits the environment. """
        assert set(THUMBSUP_BASE.keys()).issubset(self.reward_config.keys()), "Incomplete free reach reward configuration."

    def _get_achieved_goal(self):
        return np.array([1])


if __name__ == '__main__':
    # register the environment
    gym.envs.register(
        id='DexterousThumbsUp-v0',
        entry_point='thumbs-up-task:DexterousThumbsUp',
        max_episode_steps=SHADOWHAND_MAX_STEPS,
        kwargs=dict(initial_qpos=DEFAULT_INITIAL_QPOS, distance_threshold=0.02)
    )

    # create the environment
    env = make_env('DexterousThumbsUp-v0', render_mode="human")

    # run an episode
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated

    env.close()