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

        # Opponent pool (list of frozen policies, empty initially)
        self.opponent_pool = []
        if opponent_policy is not None:
            self.opponent_pool.append(opponent_policy)
            
        # The opponent chosen for the current episode
        self.current_opponent = None

        # For storing the opponent's observation
        self._opponent_obs = None

    def reset(self, seed=None, options=None):
        """Reset the environment, swap sides, and pick a new opponent."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Phase 6: Canonical Frame. Randomly assign learning agent to Green(0) or Red(1)
        self.learning_side = np.random.choice([0, 1])
        opp_side = 1 - self.learning_side

        # Sample an opponent from the pool for this episode
        if self.opponent_pool:
            self.current_opponent = np.random.choice(self.opponent_pool)
        else:
            self.current_opponent = None
            
        # Get opponent's perspective (canonical frame)
        self._opponent_obs = self.env._get_obs(team=opp_side)
        
        # Get learning agent's perspective (canonical frame)
        final_obs = self.env._get_obs(team=self.learning_side)
            
        return final_obs, info

    def step(self, action):
        """
        Step the environment.
        action: learning agent's action (canonical frame, expecting to attack right)
        """
        opp_side = 1 - self.learning_side
        
        # 1. Get opponent's canonical action (they also expect to attack right)
        if self.current_opponent is None or self._opponent_obs is None:
            canonical_opp_action = self.action_space.sample()
        else:
            canonical_opp_action, _ = self.current_opponent.predict(self._opponent_obs, deterministic=True)
            
        # 2. De-canonicalize actions into physical left/right teams
        physical_learning_action = self._mirror_action(action) if self.learning_side == 1 else action
        physical_opp_action = self._mirror_action(canonical_opp_action) if opp_side == 1 else canonical_opp_action
        
        # 3. Route actions to the physical teams
        if self.learning_side == 0:
            # Learning = Green (Left), Opp = Red (Right)
            self.env.set_red_actions(physical_opp_action)
            _, _, done, truncated, info = self.env.step(physical_learning_action)
            learning_reward = info['green_reward']
        else:
            # Learning = Red (Right), Opp = Green (Left)
            # Physical game engine expects Green's action in `step()` and Red's in `set_red_actions()`
            self.env.set_red_actions(physical_learning_action)
            _, _, done, truncated, info = self.env.step(physical_opp_action)
            learning_reward = info['red_reward']
            
        # 4. Update opponent observation for next step
        if not done and not truncated:
            self._opponent_obs = self.env._get_obs(team=opp_side)

        # 5. Canonicalize learning agent's output observation
        final_obs = self.env._get_obs(team=self.learning_side)

        # Phase 7: Expose learning side so train.py can track side-specific win rates
        info['learning_side'] = self.learning_side

        return final_obs, learning_reward, done, truncated, info

    def _mirror_obs(self, obs):
        """Mirror the observation left-right for the opponent perspective."""
        mirrored = obs.copy()
        # Mirror x-coordinates (every even index in player positions)
        for i in range(0, 12, 2):
            mirrored[i] = 1.0 - obs[i]
        # Mirror ball x (index 12)
        mirrored[12] = 1.0 - obs[12]
        # Mirror ball vx (index 14). Since vx is normalized [0, 1], 1.0 - vx flips it.
        mirrored[14] = 1.0 - obs[14]
        return mirrored

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

    def set_opponent_pool(self, pool):
        """Update the pool of available opponent policies."""
        self.opponent_pool = pool

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Clean up."""
        self.env.close()


