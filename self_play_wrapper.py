"""
Self-Play Wrapper
=================
Wraps the multi-agent FootballEnv into a single-agent Gymnasium environment
for use with Stable-Baselines3.

The wrapper controls one team (green) and uses a frozen opponent policy
for the other team (red). The opponent policy is periodically updated
with a copy of the agent's policy to enable self-play training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from football_env import FootballEnv


class SelfPlayWrapper(gym.Env):
    """
    Single-agent wrapper for multi-agent football environment.

    The learning agent controls the green team.
    The opponent (red team) is controlled by a frozen policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, opponent_policy=None):
        super().__init__()

        self.env = FootballEnv(render_mode=render_mode)
        self.render_mode = render_mode

        # Same spaces as the base environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Opponent policy (None = random actions)
        self.opponent_policy = opponent_policy

        # For storing the opponent's observation
        self._opponent_obs = None

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._opponent_obs = self.env._get_obs(team=1)
        return obs, info

    def step(self, action):
        """
        Step the environment.

        action: green team's action (6 discrete values)
        """
        # Get opponent (red team) action
        red_action = self._get_opponent_action()
        self.env.set_red_actions(red_action)

        # Step the environment with green team's action
        obs, reward, done, truncated, info = self.env.step(action)

        # Update opponent observation for next step
        if not done and not truncated:
            self._opponent_obs = self.env._get_obs(team=1)

        return obs, reward, done, truncated, info

    def _get_opponent_action(self):
        """Get action from the opponent policy."""
        if self.opponent_policy is None:
            # Random opponent
            return self.action_space.sample()

        # Use the frozen policy to get opponent actions
        # Mirror the observation so opponent sees from their perspective
        obs = self._opponent_obs
        if obs is None:
            return self.action_space.sample()

        action, _ = self.opponent_policy.predict(obs, deterministic=False)

        # Mirror the action directions for player movement and kicks
        # Since red team attacks left, we need to mirror horizontal directions
        mirrored_action = self._mirror_action(action)
        return mirrored_action

    def _mirror_action(self, action):
        """
        Mirror action directions for the opponent.
        Directions: 0=stay, 1=up, 2=up-right, 3=right, 4=down-right,
                    5=down, 6=down-left, 7=left, 8=up-left

        Mirror mapping (flip left/right):
        0->0, 1->1, 2->8, 3->7, 4->6, 5->5, 6->4, 7->3, 8->2
        """
        mirror_map = {0: 0, 1: 1, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2}
        mirrored = np.array(action, dtype=np.int64)
        for i in range(0, len(mirrored), 2):
            mirrored[i] = mirror_map.get(int(mirrored[i]), int(mirrored[i]))
        for i in range(1, len(mirrored), 2):
            mirrored[i] = mirror_map.get(int(mirrored[i]), int(mirrored[i]))
        return mirrored

    def set_opponent_policy(self, policy):
        """Update the opponent's policy (for self-play updates)."""
        self.opponent_policy = policy

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Clean up."""
        self.env.close()


class MirroredObsWrapper(gym.ObservationWrapper):
    """
    Wrapper that mirrors observations for the red team perspective.
    This ensures the opponent policy sees the pitch from its own viewpoint.
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        """Mirror the observation left-right."""
        mirrored = obs.copy()
        # Mirror x-coordinates (every even index in player positions)
        for i in range(0, 12, 2):
            mirrored[i] = 1.0 - obs[i]
        # Mirror ball x
        mirrored[12] = 1.0 - obs[12]
        # Mirror ball vx
        mirrored[14] = 1.0 - obs[14]
        return mirrored
