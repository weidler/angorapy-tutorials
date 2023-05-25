from angorapy.common.const import VISION_WH
from angorapy.common.senses import Sensation
from angorapy.environments.hand.shadowhand import BaseShadowHandEnv
import numpy as np
from angorapy.environments.utils import robot_get_obs


class DexterousThumbsUp(BaseShadowHandEnv):
    """A task where the agent has to raise or low its thumb based on a goal provided as a boolean flag."""

    def __init__(self, initial_qpos, distance_threshold):
        super().__init__(initial_qpos, distance_threshold)

        self._thumb_tip = self.model.site("robot0:S_thtip")

    def _set_default_reward_function_and_config(self):
        self.reward_function = thumbsup
        self.reward_config = THUMBSUP_BASE

    def _sample_goal(self) -> np.array:
        """Sample a new goal and return it. The goal is a boolean flag indicating whether the thumb should be raised or
        lowered. A 1 indicates that the thumb should be raised, a 0 indicates that the thumb should be lowered."""
        return np.random.binomial(1, 0.5, size=[1])

    def _get_obs(self) -> np.array:
        touch = self.data.sensordata[self._touch_sensor_id]

        achieved_goal = self._get_achieved_goal().ravel()

        robot_qpos, robot_qvel = robot_get_obs(self.model, self.data)
        proprioception = np.concatenate([robot_qpos, robot_qvel])

        return {
            'observation': Sensation(
                proprioception=proprioception,
                touch=touch if self.touch else None,
                vision=self.render("rgb_array", VISION_WH, VISION_WH) if self.vision else np.array([]),
                goal=self.goal.copy()
            ),

            'desired_goal': self.goal.copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def step(self, action: np.ndarray):
        """Perform one step of the environment's dynamics. When the goal is achieved, the episode terminates."""
        obs, reward, terminated, truncated, info = super().step(action)
        done = (terminated or truncated)

        if done:
            info['success'] = int(np.allclose(obs['achieved_goal'], obs['desired_goal']))

        return obs, reward, done, info
