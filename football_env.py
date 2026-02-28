"""
Football Simulator Environment
==============================
A 2v2+GK (3v3) football environment built with Gymnasium.
Implements comprehensive football rules in a scaled-down pitch.

Teams:
  - Green team (left side)
  - Red team (right side)
  - 2 outfield players + 1 goalkeeper per team

Rules implemented:
  - Kickoff from center after each goal
  - First to 2 goals wins
  - Throw-in when ball crosses sidelines
  - Goal kick when ball crosses end-line without goal
  - Corner kick when defending team puts ball over own end-line
  - Goalkeeper can only handle ball in their penalty box
  - Offside (simplified: attacker can't be behind last defender when ball is played forward)
  - Fouls: overly aggressive tackles result in free kicks
  - Free kicks from foul location
  - Ball possession and dribbling mechanics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# ============================================================
# CONSTANTS
# ============================================================

# Pitch dimensions (pixels)
PITCH_WIDTH = 800
PITCH_HEIGHT = 500
PITCH_MARGIN = 40  # margin around the pitch for rendering

# Goal dimensions
GOAL_WIDTH = 10  # depth of the goal
GOAL_HEIGHT = 120  # height of the goal opening
GOAL_Y_TOP = (PITCH_HEIGHT - GOAL_HEIGHT) / 2
GOAL_Y_BOTTOM = (PITCH_HEIGHT + GOAL_HEIGHT) / 2

# Penalty box dimensions
PENALTY_BOX_WIDTH = 100
PENALTY_BOX_HEIGHT = 200
PENALTY_BOX_Y_TOP = (PITCH_HEIGHT - PENALTY_BOX_HEIGHT) / 2
PENALTY_BOX_Y_BOTTOM = (PITCH_HEIGHT + PENALTY_BOX_HEIGHT) / 2

# Goal area (6-yard box)
GOAL_AREA_WIDTH = 50
GOAL_AREA_HEIGHT = 140
GOAL_AREA_Y_TOP = (PITCH_HEIGHT - GOAL_AREA_HEIGHT) / 2
GOAL_AREA_Y_BOTTOM = (PITCH_HEIGHT + GOAL_AREA_HEIGHT) / 2

# Center circle
CENTER_CIRCLE_RADIUS = 60

# Player properties
PLAYER_RADIUS = 12
PLAYER_MAX_SPEED = 5.0  # Increased for faster gameplay
GK_MAX_SPEED = 4.5      # Goalkeepers slightly slower
PLAYER_KICK_RANGE = 20  # distance at which player can kick ball
PLAYER_TACKLE_RANGE = 18  # distance for tackling

# Ball properties
BALL_RADIUS = 6
BALL_MAX_SPEED = 10.0
BALL_FRICTION = 0.97  # friction coefficient per frame
BALL_KICK_POWER = 8.0
BALL_PASS_POWER = 6.0

# Game settings
MAX_STEPS = 3000  # max steps per episode
GOALS_TO_WIN = 2

# Movement directions (8 directions + stay)
# Index: 0=stay, 1=up, 2=up-right, 3=right, 4=down-right, 5=down, 6=down-left, 7=left, 8=up-left
DIRECTION_VECTORS = np.array([
    [0, 0],       # 0: stay
    [0, -1],      # 1: up
    [0.707, -0.707],  # 2: up-right
    [1, 0],       # 3: right
    [0.707, 0.707],   # 4: down-right
    [0, 1],       # 5: down
    [-0.707, 0.707],  # 6: down-left
    [-1, 0],      # 7: left
    [-0.707, -0.707], # 8: up-left
], dtype=np.float32)

# Kick directions (same 8 directions + no kick)
KICK_DIRECTION_VECTORS = DIRECTION_VECTORS.copy()

# Game states
class GameState:
    PLAYING = 0
    KICKOFF = 1
    THROW_IN = 2
    GOAL_KICK = 3
    CORNER_KICK = 4
    FREE_KICK = 5
    GOAL_SCORED = 6


class Player:
    """Represents a player on the pitch."""

    def __init__(self, team, player_id, is_goalkeeper=False):
        self.team = team  # 0 = green (left), 1 = red (right)
        self.player_id = player_id
        self.is_goalkeeper = is_goalkeeper
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.has_ball = False
        self.max_speed = GK_MAX_SPEED if is_goalkeeper else PLAYER_MAX_SPEED
        self.stamina = 1.0  # 0-1, affects speed
        self.cooldown = 0  # action cooldown (for tackles/kicks)

    def reset_position(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.has_ball = False
        self.stamina = 1.0
        self.cooldown = 0

    @property
    def pos(self):
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


class Ball:
    """Represents the football."""

    def __init__(self):
        self.x = PITCH_WIDTH / 2
        self.y = PITCH_HEIGHT / 2
        self.vx = 0.0
        self.vy = 0.0
        self.owner = None  # Player object or None
        self.last_touch_team = -1  # -1 = none, 0 = green, 1 = red
        self.last_kicker = None
        self.second_last_touch_team = -1
        self.kicked_from_x = 0.0

    def reset(self, x=None, y=None):
        self.x = x if x is not None else PITCH_WIDTH / 2
        self.y = y if y is not None else PITCH_HEIGHT / 2
        self.vx = 0.0
        self.vy = 0.0
        self.owner = None
        self.last_touch_team = -1
        self.last_kicker = None  # Track the player who last kicked the ball
        self.second_last_touch_team = -1  # Track the team that touched the ball before the last kicker
        self.kicked_from_x = 0.0  # Track x coordinate where the ball was passed from

    @property
    def pos(self):
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def speed(self):
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    def update(self):
        """Update ball position with friction."""
        if self.owner is not None:
            # Ball follows the player who has it
            self.x = self.owner.x + (PLAYER_RADIUS + BALL_RADIUS + 2) * \
                     (1 if self.owner.team == 0 else -1)
            self.y = self.owner.y
            self.vx = 0
            self.vy = 0
        else:
            self.x += self.vx
            self.y += self.vy
            self.vx *= BALL_FRICTION
            self.vy *= BALL_FRICTION
            # Stop ball if very slow
            if abs(self.vx) < 0.1 and abs(self.vy) < 0.1:
                self.vx = 0
                self.vy = 0


class FootballEnv(gym.Env):
    """
    Football Simulator Gymnasium Environment.

    Multi-agent environment where two teams of 3 players each (2 outfield + 1 GK)
    compete to score 2 goals first.

    Observation Space (per team, 18 values):
        - Own 3 players' (x, y) = 6
        - Opponent 3 players' (x, y) = 6
        - Ball (x, y, vx, vy) = 4
        - Scores (own, opponent) = 2
        Total = 18

    Action Space (per team, MultiDiscrete):
        For each of 3 players: [movement_dir (9), kick_dir (9)]
        Total = 6 discrete values
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None

        # Action space: 3 players × (move_dir, kick_dir)
        # Each player has: movement (9 choices) + kick/action (9 choices)
        self.action_space = spaces.MultiDiscrete([9, 9, 9, 9, 9, 9])

        # Observation space: normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )

        # Initialize teams
        self.green_team = [
            Player(0, 0, is_goalkeeper=True),   # GK
            Player(0, 1, is_goalkeeper=False),   # Outfield 1
            Player(0, 2, is_goalkeeper=False),   # Outfield 2
        ]
        self.red_team = [
            Player(1, 0, is_goalkeeper=True),    # GK
            Player(1, 1, is_goalkeeper=False),    # Outfield 1
            Player(1, 2, is_goalkeeper=False),    # Outfield 2
        ]
        self.all_players = self.green_team + self.red_team

        # Ball
        self.ball = Ball()

        # Scores
        self.green_score = 0
        self.red_score = 0

        # Game state
        self.game_state = GameState.KICKOFF
        self.kickoff_team = 0  # which team kicks off
        self.set_piece_team = 0  # team taking set piece
        self.set_piece_pos = (0, 0)  # position for set piece
        self.state_timer = 0  # timer for pauses between states
        self.steps = 0
        self.done = False

        # Offside tracking
        self.last_pass_positions = {}  # track positions when pass was made

        # Stats for reward computation
        self.prev_ball_x = PITCH_WIDTH / 2
        self.prev_green_score = 0
        self.prev_red_score = 0

        # Match statistics and event rewards
        self.match_stats = self._empty_stats()
        self._event_rewards = 0.0

    def _empty_stats(self):
        return {
            "green_possession_steps": 0,
            "red_possession_steps": 0,
            "total_steps": 0,
            "green_passes": 0,
            "red_passes": 0,
            "green_shots": 0,
            "red_shots": 0,
            "green_saves": 0,
            "red_saves": 0,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new match."""
        super().reset(seed=seed)

        self.green_score = 0
        self.red_score = 0
        self.steps = 0
        self.done = False
        self.game_state = GameState.KICKOFF
        self.kickoff_team = 0
        self.state_timer = 0
        self.current_episode = options.get("episode", 0) if options else 0

        # Reset ball to center
        self.ball.reset()

        # Place players in starting positions
        self._place_players_kickoff()

        self.prev_ball_x = self.ball.x
        self.prev_green_score = 0
        self.prev_red_score = 0

        self.match_stats = self._empty_stats()
        self._event_rewards = 0.0

        obs = self._get_obs(team=0)
        info = self._get_info()
        return obs, info

    def _place_players_kickoff(self):
        """Place all players in kickoff formation."""
        # Green team (left side)
        self.green_team[0].reset_position(30, PITCH_HEIGHT / 2)   # GK
        self.green_team[1].reset_position(PITCH_WIDTH / 2 - 80, PITCH_HEIGHT / 2 - 60)  # Outfield 1
        self.green_team[2].reset_position(PITCH_WIDTH / 2 - 80, PITCH_HEIGHT / 2 + 60)  # Outfield 2

        # Red team (right side)
        self.red_team[0].reset_position(PITCH_WIDTH - 30, PITCH_HEIGHT / 2)  # GK
        self.red_team[1].reset_position(PITCH_WIDTH / 2 + 80, PITCH_HEIGHT / 2 - 60)  # Outfield 1
        self.red_team[2].reset_position(PITCH_WIDTH / 2 + 80, PITCH_HEIGHT / 2 + 60)  # Outfield 2

    def _place_players_restart(self, restart_type, team, pos):
        """Place players appropriately for a restart."""
        # For set pieces, we keep players roughly where they are
        # but ensure minimum distances from the ball
        min_dist = 50  # minimum distance from ball for opposing team
        for player in self.all_players:
            if player.team != team:
                dist = player.distance_to(pos[0], pos[1])
                if dist < min_dist:
                    # Push player away
                    dx = player.x - pos[0]
                    dy = player.y - pos[1]
                    length = max(math.sqrt(dx*dx + dy*dy), 0.1)
                    player.x = pos[0] + (dx / length) * min_dist
                    player.y = pos[1] + (dy / length) * min_dist

        # Keep players in bounds
        for player in self.all_players:
            player.x = np.clip(player.x, PLAYER_RADIUS, PITCH_WIDTH - PLAYER_RADIUS)
            player.y = np.clip(player.y, PLAYER_RADIUS, PITCH_HEIGHT - PLAYER_RADIUS)

    def step(self, action):
        """
        Execute one step.

        action: array of 6 ints — [move0, kick0, move1, kick1, move2, kick2]
                for the green team's 3 players.
                Red team actions are provided via the self-play wrapper.
        """
        if self.done:
            obs = self._get_obs(team=0)
            return obs, 0.0, True, False, self._get_info()

        self.steps += 1

        # Decode actions for green team
        green_actions = self._decode_actions(action)

        # Red team actions (stored by self-play wrapper, or random)
        red_actions = self._red_team_actions

        # Handle set piece states
        if self.game_state != GameState.PLAYING:
            self._update_cooldowns()
            self._update_stamina()
            self.state_timer += 1
            if self.state_timer >= 15:  # Brief pause before restart
                self._execute_set_piece()
                self.game_state = GameState.PLAYING
                self.state_timer = 0
        else:
            # ---- PROCESS ACTIONS ----
            # 1. Move players
            self._move_players(self.green_team, green_actions)
            self._move_players(self.red_team, red_actions)

            # 2. Handle ball-player interactions
            self._handle_ball_interactions()

            # 3. Process kicks
            self._process_kicks(self.green_team, green_actions)
            if self.game_state != GameState.PLAYING:
                self.ball.vx = 0
                self.ball.vy = 0
            else:
                self._process_kicks(self.red_team, red_actions)

            # 4. Update ball physics
            self.ball.update()

            # 5. Check rules
            self._check_goals()
            self._check_out_of_bounds()
            self._enforce_goalkeeper_restrictions()
            self._update_cooldowns()
            self._update_stamina()

        # Update possession stats
        self.match_stats["total_steps"] += 1
        if self.ball.owner is not None:
            if self.ball.owner.team == 0:
                self.match_stats["green_possession_steps"] += 1
            else:
                self.match_stats["red_possession_steps"] += 1

        # Check win condition
        truncated = False
        if self.green_score >= GOALS_TO_WIN or self.red_score >= GOALS_TO_WIN:
            self.done = True
        elif self.steps >= MAX_STEPS:
            truncated = True
            self.done = True

        # Compute reward for both teams
        self.truncated = truncated
        green_reward = self._compute_reward(team=0)
        red_reward = self._compute_reward(team=1)
        
        # Add discrete event rewards (which are tracked from Green's perspective)
        green_reward += self._event_rewards
        red_reward -= self._event_rewards
        self._event_rewards = 0.0

        # Update tracking
        self.prev_ball_x = self.ball.x
        self.prev_green_score = self.green_score
        self.prev_red_score = self.red_score

        obs = self._get_obs(team=0)
        info = self._get_info()
        
        # Inject both rewards into info for wrapper to harvest
        info['green_reward'] = green_reward
        info['red_reward'] = red_reward

        # Render if needed
        if self.render_mode == "human":
            self.render()

        # By default we return Green's reward to SB3 (unless the wrapper swaps it)
        return obs, green_reward, self.done, truncated, info

    def set_red_actions(self, action):
        """Set the red team's actions (called by self-play wrapper)."""
        self._red_team_actions = self._decode_actions(action)

    @property
    def _red_team_actions(self):
        if not hasattr(self, '_red_actions_cache'):
            # Default: random actions
            self._red_actions_cache = [
                (self.np_random.integers(0, 9), self.np_random.integers(0, 9))
                for _ in range(3)
            ]
        return self._red_actions_cache

    @_red_team_actions.setter
    def _red_team_actions(self, value):
        self._red_actions_cache = value

    def _decode_actions(self, action):
        """Decode flat action array into list of (move_dir, kick_dir) tuples."""
        return [
            (int(action[0]), int(action[1])),  # Player 0 (GK)
            (int(action[2]), int(action[3])),  # Player 1
            (int(action[4]), int(action[5])),  # Player 2
        ]

    def _move_players(self, team, actions):
        """Move players based on their movement actions."""
        for i, player in enumerate(team):
            move_dir = actions[i][0]
            if move_dir == 0:
                # Decelerate
                player.vx *= 0.5
                player.vy *= 0.5
                continue

            direction = DIRECTION_VECTORS[move_dir]
            speed = player.max_speed * player.stamina

            player.vx = direction[0] * speed
            player.vy = direction[1] * speed

            new_x = player.x + player.vx
            new_y = player.y + player.vy

            # Keep within pitch bounds
            new_x = np.clip(new_x, PLAYER_RADIUS, PITCH_WIDTH - PLAYER_RADIUS)
            new_y = np.clip(new_y, PLAYER_RADIUS, PITCH_HEIGHT - PLAYER_RADIUS)

            # Goalkeeper movement restriction (can't go past halfway)
            if player.is_goalkeeper:
                if player.team == 0:  # Green GK stays in left half
                    new_x = np.clip(new_x, PLAYER_RADIUS, PITCH_WIDTH / 2 - 20)
                else:  # Red GK stays in right half
                    new_x = np.clip(new_x, PITCH_WIDTH / 2 + 20, PITCH_WIDTH - PLAYER_RADIUS)

            # Player-player collision
            for other in self.all_players:
                if other is player:
                    continue
                dist = math.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
                if dist < PLAYER_RADIUS * 2:
                    # Push apart
                    overlap = PLAYER_RADIUS * 2 - dist
                    if dist > 0:
                        dx = (new_x - other.x) / dist
                        dy = (new_y - other.y) / dist
                        new_x += dx * overlap / 2
                        new_y += dy * overlap / 2

            player.x = float(np.clip(new_x, PLAYER_RADIUS, PITCH_WIDTH - PLAYER_RADIUS))
            player.y = float(np.clip(new_y, PLAYER_RADIUS, PITCH_HEIGHT - PLAYER_RADIUS))

    def _handle_ball_interactions(self):
        """Handle automatic ball pickup when players are close."""
        if self.ball.owner is not None:
            return

        # Handle approximate deflection ownership before pickup
        if self.ball.owner is None and self.ball.speed > 1.5:
            allow_deflection_mark = True
            if self.ball.last_kicker is not None:
                if self.ball.last_kicker.distance_to(self.ball.x, self.ball.y) < PLAYER_KICK_RANGE:
                    allow_deflection_mark = False

            if allow_deflection_mark:
                for p in self.all_players:
                    # Teammates shouldn't "deflect" their own pass
                    if self.ball.last_kicker is not None and p.team == self.ball.last_kicker.team:
                        continue
                    if p.distance_to(self.ball.x, self.ball.y) < (PLAYER_RADIUS + BALL_RADIUS - 2):
                        self.ball.last_touch_team = p.team
                        break

        closest_player = None
        closest_dist = float('inf')

        for player in self.all_players:
            if player.cooldown > 0:
                continue
            # Check if goalkeeper is trying to handle ball outside the box
            if player.is_goalkeeper and not self._is_in_penalty_box(player):
                # GK can still use feet outside box, but no handling possession
                continue
                
            dist = player.distance_to(self.ball.x, self.ball.y)
            if dist < PLAYER_KICK_RANGE and dist < closest_dist:
                closest_dist = dist
                closest_player = player

        if closest_player is not None:
            # 1. Successful Pass Detection
            if (self.ball.last_kicker is not None and 
                self.ball.last_kicker.team == closest_player.team and 
                self.ball.last_kicker is not closest_player):
                
                # It's a pass!
                if closest_player.team == 0:
                    self.match_stats["green_passes"] += 1
                    pass_reward = 0.1  # Base pass
                    
                    # Evaluate progression and key passes
                    # Green attacks Right (larger X is better)
                    if self.ball.x > self.ball.kicked_from_x + 50.0:
                        pass_reward += 0.1  # Progressive pass
                    if self.ball.x > PITCH_WIDTH * 0.66:
                        pass_reward += 0.2  # Key pass
                        
                    self._event_rewards += pass_reward
                else:
                    self.match_stats["red_passes"] += 1
                    pass_reward = 0.1
                    
                    # Red attacks Left (smaller X is better)
                    if self.ball.x < self.ball.kicked_from_x - 50.0:
                        pass_reward += 0.1
                    if self.ball.x < PITCH_WIDTH * 0.33:
                        pass_reward += 0.2
                        
                    self._event_rewards -= pass_reward

            # 2. Goalkeeper Save Detection
            # If GK picks up a fast-moving ball last kicked by the opponent
            is_shot_on_goal = False
            # Check if ball is roughly heading for the goal mouth area
            is_near_goal_height = (GOAL_Y_TOP - 30 <= self.ball.y <= GOAL_Y_BOTTOM + 30)
            
            if closest_player.team == 0:
                # Green GK (Left goal): ball moving left (vx < -1), left third, correct height
                if self.ball.vx < -1.0 and self.ball.x < PITCH_WIDTH * 0.35 and is_near_goal_height:
                    is_shot_on_goal = True
            else:
                # Red GK (Right goal): ball moving right (vx > 1), right third, correct height
                if self.ball.vx > 1.0 and self.ball.x > PITCH_WIDTH * 0.65 and is_near_goal_height:
                    is_shot_on_goal = True

            if (closest_player.is_goalkeeper and self.ball.speed > 3.0 and
                is_shot_on_goal and
                self.ball.last_kicker is not None and 
                self.ball.last_kicker.team != closest_player.team):
                
                if closest_player.team == 0:
                    self.match_stats["green_saves"] += 1
                    self._event_rewards += 2.0  # Normalized save reward
                else:
                    self.match_stats["red_saves"] += 1
                    self._event_rewards -= 2.0  # -2 penalty if red GK saves our shot

            # 3. Turnover Detection
            elif (self.ball.last_touch_team != -1 and 
                  self.ball.last_touch_team != closest_player.team and
                  not closest_player.is_goalkeeper):
                # Opponent just lost the ball to us, or we lost it to them
                if closest_player.team == 0:
                    # Red lost the ball. Green gets interception/recovery
                    turnover_value = 0.2
                    
                    if self.ball.x > PITCH_WIDTH * 0.66: # High press
                        turnover_value += 0.3
                    self._event_rewards += turnover_value 
                else:
                    # Green lost the ball. Red gets interception/recovery
                    turnover_value = 0.2
                    
                    if self.ball.x < PITCH_WIDTH * 0.33: # High press
                        turnover_value += 0.3
                    self._event_rewards -= turnover_value
                
                # Turnovers break the pass/assist chain
                self.ball.second_last_touch_team = -1

            # Give possession to closest player
            self.ball.owner = closest_player
            closest_player.has_ball = True
            self.ball.last_touch_team = closest_player.team

            # Once the ball is received, last_kicker should be cleared
            self.ball.last_kicker = None
            self.ball.kicked_from_x = 0.0
            # second_last_touch_team is preserved for assist logic until a goal or restart


    def _process_kicks(self, team, actions):
        """Process kick actions for a team."""
        for i, player in enumerate(team):
            kick_dir = actions[i][1]
            if kick_dir == 0:  # no kick
                continue
            if not player.has_ball and player is not self.ball.owner:
                # Can still attempt tackle if close enough
                if player.cooldown <= 0:
                    self._attempt_tackle(player)
                continue
            if player.cooldown > 0:
                continue

            # Perform kick
            direction = KICK_DIRECTION_VECTORS[kick_dir]
            power = BALL_KICK_POWER
            
            is_forward = (player.team == 0 and direction[0] > 0.2) or (player.team == 1 and direction[0] < -0.2)

            # Check if this is a pass (teammate in kick direction)
            is_pass = self._check_if_pass(player, direction, team)
            if is_pass:
                power = BALL_PASS_POWER

            # Release ball
            player.has_ball = False
            self.ball.owner = None
            self.ball.vx = direction[0] * power
            self.ball.vy = direction[1] * power
            self.ball.second_last_touch_team = self.ball.last_touch_team
            self.ball.last_touch_team = player.team
            self.ball.last_kicker = player
            self.ball.kicked_from_x = player.x  # Record where the ball was passed/shot from
            player.cooldown = 5  # brief cooldown after kicking
            
            if is_forward:
                self._check_offside_on_kick(kicker=player, direction=direction)
                if self.game_state != GameState.PLAYING:
                    return

            # Shot statistics tracking
            # If kicked towards the opponent's goal mouth
            if power >= BALL_KICK_POWER * 0.9:
                if player.team == 0 and direction[0] > 0.5:
                    if GOAL_Y_TOP - 40 <= self.ball.y <= GOAL_Y_BOTTOM + 40:
                        self.match_stats["green_shots"] += 1
                        
                        # xG Calculation for Green
                        dist = math.sqrt((PITCH_WIDTH - player.x)**2 + (PITCH_HEIGHT/2 - player.y)**2)
                        angle_penalty = abs(PITCH_HEIGHT/2 - player.y) / (PITCH_HEIGHT/2)
                        xg = max(0.01, 1.0 - (dist * 0.002) - (angle_penalty * 0.3))
                        self._event_rewards += (0.5 * xg)
                        self._event_rewards += 0.2  # Shot on target
                        
                elif player.team == 1 and direction[0] < -0.5:
                    if GOAL_Y_TOP - 40 <= self.ball.y <= GOAL_Y_BOTTOM + 40:
                        self.match_stats["red_shots"] += 1
                        
                        # xG Calculation for Red
                        dist = math.sqrt((player.x - 0)**2 + (PITCH_HEIGHT/2 - player.y)**2)
                        angle_penalty = abs(PITCH_HEIGHT/2 - player.y) / (PITCH_HEIGHT/2)
                        xg = max(0.01, 1.0 - (dist * 0.002) - (angle_penalty * 0.3))
                        self._event_rewards -= (0.5 * xg)
                        self._event_rewards -= 0.2  # Shot on target penalty

    def _attempt_tackle(self, player):
        """Attempt to tackle the ball carrier."""
        if self.ball.owner is None:
            return

        target = self.ball.owner
        if target.team == player.team:
            return  # can't tackle own team

        dist = player.distance_to(target.x, target.y)
        if dist > PLAYER_TACKLE_RANGE:
            return

        # Tackle success probability based on distance and stamina
        success_chance = 0.15 + 0.55 * player.stamina * (1 - dist / PLAYER_TACKLE_RANGE)
        success_chance = float(np.clip(success_chance, 0.05, 0.75))

        if self.np_random.random() < success_chance:
            # Successful tackle
            target.has_ball = False
            self.ball.owner = player
            player.has_ball = True
            self.ball.last_touch_team = player.team
            player.cooldown = 8

            # Clear kick context on successful tackle to prevent false pass/save credit
            self.ball.last_kicker = None
            self.ball.kicked_from_x = 0.0
            self.ball.second_last_touch_team = -1
        else:
            # Failed tackle — foul!
            foul_chance = 0.4 * (1 - player.stamina)
            if self.np_random.random() < foul_chance:
                self._call_foul(player, target)
            player.cooldown = 12  # longer cooldown for failed tackle

    def _call_foul(self, fouling_player, fouled_player):
        """Handle a foul — award free kick to opposing team."""
        self.game_state = GameState.FREE_KICK
        self.set_piece_team = fouled_player.team
        self.set_piece_pos = (fouled_player.x, fouled_player.y)
        self.state_timer = 0

        # Release ball
        if self.ball.owner is not None:
            self.ball.owner.has_ball = False
        self.ball.owner = None
        self.ball.x = fouled_player.x
        self.ball.y = fouled_player.y
        self.ball.vx = 0
        self.ball.vy = 0
        self.ball.last_touch_team = -1
        self.ball.second_last_touch_team = -1
        self.ball.last_kicker = None
        self.ball.kicked_from_x = 0.0
        
        # Zero out player velocities for clean restart
        for p in self.all_players:
            p.vx = 0.0
            p.vy = 0.0

    def _check_if_pass(self, player, direction, team):
        """Check if a kick direction would be a pass to a teammate."""
        for teammate in team:
            if teammate is player:
                continue
            dx = teammate.x - player.x
            dy = teammate.y - player.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 30:
                continue  # too close
            # Check if teammate is roughly in the kick direction
            if dist > 0:
                to_mate = np.array([dx/dist, dy/dist])
                dot_product = direction[0] * to_mate[0] + direction[1] * to_mate[1]
                if dot_product > 0.7:  # within ~45 degrees
                    return True
        return False

    def _check_goals(self):
        """Check if a goal has been scored."""
        if self.game_state != GameState.PLAYING:
            return

        ball_x = self.ball.x
        ball_y = self.ball.y

        # Goal for red scored by green (ball crosses right end-line in goal area)
        if ball_x >= PITCH_WIDTH - BALL_RADIUS:
            if GOAL_Y_TOP <= ball_y <= GOAL_Y_BOTTOM:
                self.green_score += 1
                self._handle_goal_scored(scoring_team=0)
                return

        # Goal for green scored by red (ball crosses left end-line in goal area)
        if ball_x <= BALL_RADIUS:
            if GOAL_Y_TOP <= ball_y <= GOAL_Y_BOTTOM:
                self.red_score += 1
                self._handle_goal_scored(scoring_team=1)
                return

    def _handle_goal_scored(self, scoring_team):
        """Handle goal scored — reset for kickoff."""
        self.game_state = GameState.KICKOFF
        # Kickoff goes to the team that conceded
        self.kickoff_team = 1 - scoring_team
        self.state_timer = 0  # Start from 0 to allow pause up to 15

        # Zero out player velocities for clean restart
        for p in self.all_players:
            p.vx = 0.0
            p.vy = 0.0

    def _clear_ball_context(self):
        """Reset ball ownership and pass/kick context for a clean restart."""
        self.ball.last_touch_team = -1
        self.ball.second_last_touch_team = -1
        self.ball.last_kicker = None
        self.ball.kicked_from_x = 0.0
        self.ball.vx = 0
        self.ball.vy = 0
        if self.ball.owner:
            self.ball.owner.has_ball = False
        self.ball.owner = None

    def _check_out_of_bounds(self):
        """Check if ball has gone out of bounds and handle restarts."""
        if self.game_state != GameState.PLAYING:
            return

        ball_x = self.ball.x
        ball_y = self.ball.y
        last_touch = self.ball.last_touch_team  # snapshot

        # Side-line out (top or bottom)
        if ball_y <= BALL_RADIUS or ball_y >= PITCH_HEIGHT - BALL_RADIUS:
            # Side-line out -> Throw-in
            if last_touch == -1:
                nearest = min(self.all_players, key=lambda p: p.distance_to(ball_x, ball_y))
                self.set_piece_team = nearest.team
            else:
                self.set_piece_team = 1 - last_touch
                
            self._clear_ball_context()
            self.game_state = GameState.THROW_IN
            # Throw-in from where ball went out
            throw_x = np.clip(ball_x, 10, PITCH_WIDTH - 10)
            throw_y = BALL_RADIUS + 5 if ball_y <= BALL_RADIUS else PITCH_HEIGHT - BALL_RADIUS - 5
            self.set_piece_pos = (throw_x, throw_y)
            self.state_timer = 0
            
            # Zero out player velocities for clean restart
            for p in self.all_players:
                p.vx = 0.0
                p.vy = 0.0
            return

        # End-line out (left or right) — not a goal
        if ball_x <= BALL_RADIUS:
            # If it's inside the goal mouth, let _check_goals handle it
            if GOAL_Y_TOP <= ball_y <= GOAL_Y_BOTTOM:
                return
            
            if last_touch == -1:
                # Default: goal kick for green
                self.game_state = GameState.GOAL_KICK
                self.set_piece_team = 0
                self.set_piece_pos = (GOAL_AREA_WIDTH, PITCH_HEIGHT / 2)
            elif last_touch == 0:
                # Green touched last -> corner for red
                self.game_state = GameState.CORNER_KICK
                self.set_piece_team = 1
                corner_y = BALL_RADIUS + 5 if ball_y < PITCH_HEIGHT / 2 else PITCH_HEIGHT - BALL_RADIUS - 5
                self.set_piece_pos = (BALL_RADIUS + 5, corner_y)
            else:
                # Red touched last -> goal kick for green
                self.game_state = GameState.GOAL_KICK
                self.set_piece_team = 0
                self.set_piece_pos = (GOAL_AREA_WIDTH, PITCH_HEIGHT / 2)

            self._clear_ball_context()
            self.state_timer = 0
            # Zero out player velocities for clean restart
            for p in self.all_players:
                p.vx = 0.0
                p.vy = 0.0
            return

        if ball_x >= PITCH_WIDTH - BALL_RADIUS:
            # If it's inside the goal mouth, let _check_goals handle it
            if GOAL_Y_TOP <= ball_y <= GOAL_Y_BOTTOM:
                return
                
            if last_touch == -1:
                # Default: goal kick for red
                self.game_state = GameState.GOAL_KICK
                self.set_piece_team = 1
                self.set_piece_pos = (PITCH_WIDTH - GOAL_AREA_WIDTH, PITCH_HEIGHT / 2)
            elif last_touch == 1:
                # Red touched last -> corner for green
                self.game_state = GameState.CORNER_KICK
                self.set_piece_team = 0
                corner_y = BALL_RADIUS + 5 if ball_y < PITCH_HEIGHT / 2 else PITCH_HEIGHT - BALL_RADIUS - 5
                self.set_piece_pos = (PITCH_WIDTH - BALL_RADIUS - 5, corner_y)
            else:
                # Green touched last -> goal kick for red
                self.game_state = GameState.GOAL_KICK
                self.set_piece_team = 1
                self.set_piece_pos = (PITCH_WIDTH - GOAL_AREA_WIDTH, PITCH_HEIGHT / 2)

            self._clear_ball_context()
            self.state_timer = 0
            for p in self.all_players:
                p.vx = 0.0
                p.vy = 0.0
            return

    def _check_offside_on_kick(self, kicker, direction):
        """
        Simplified offside rule:
        A player is offside if they are in the opponent's half and behind
        the last defender (excluding GK) exactly when a teammate plays the ball forward.
        """
        if self.game_state != GameState.PLAYING:
            return

        if kicker.team == 0:  # Green attacks right
            attacking_team = self.green_team
            defending_team = self.red_team
            forward_dir = 1
        else:  # Red attacks left
            attacking_team = self.red_team
            defending_team = self.green_team
            forward_dir = -1

        # Find last defender's x position (excluding GK)
        defenders_x = [p.x for p in defending_team if not p.is_goalkeeper]
        if not defenders_x:
            return

        last_defender_x = max(defenders_x) if forward_dir == 1 else min(defenders_x)

        for attacker in attacking_team:
            if attacker is kicker or attacker.is_goalkeeper:
                continue

            in_opp_half = attacker.x > PITCH_WIDTH / 2 if forward_dir == 1 else attacker.x < PITCH_WIDTH / 2
            beyond_line = attacker.x > last_defender_x if forward_dir == 1 else attacker.x < last_defender_x
            ahead_of_ball = attacker.x > kicker.x if forward_dir == 1 else attacker.x < kicker.x

            dx = attacker.x - kicker.x
            dy = attacker.y - kicker.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                to_att = np.array([dx / dist, dy / dist])
                if to_att[0] * direction[0] + to_att[1] * direction[1] < 0.5:
                    continue
            else:
                # Kicker and attacker are effectively on top of each other
                continue

            if in_opp_half and beyond_line and ahead_of_ball:
                # Player is offside — award free kick to defending team
                self.game_state = GameState.FREE_KICK
                self.set_piece_team = 1 - kicker.team
                self.set_piece_pos = (attacker.x, attacker.y)
                self.state_timer = 0
                
                if self.ball.owner:
                    self.ball.owner.has_ball = False
                self.ball.owner = None
                self.ball.x, self.ball.y = self.set_piece_pos
                self.ball.vx = 0
                self.ball.vy = 0
                self.ball.last_touch_team = -1
                self.ball.second_last_touch_team = -1
                self.ball.last_kicker = None
                self.ball.kicked_from_x = 0.0
                
                # Zero out player velocities for clean restart
                for p in self.all_players:
                    p.vx = 0.0
                    p.vy = 0.0
                return

    def _enforce_goalkeeper_restrictions(self):
        """Ensure goalkeepers only handle the ball inside their penalty box."""
        for team in [self.green_team, self.red_team]:
            gk = team[0]
            if gk.has_ball and not self._is_in_penalty_box(gk):
                # GK must release the ball if outside penalty box
                # They can still kick it, but not hold it
                gk.has_ball = False
                self.ball.owner = None
                # Give ball a small push in the direction GK was moving
                if abs(gk.vx) > 0.1 or abs(gk.vy) > 0.1:
                    speed = math.sqrt(gk.vx**2 + gk.vy**2)
                    self.ball.vx = (gk.vx / speed) * BALL_PASS_POWER * 0.5
                    self.ball.vy = (gk.vy / speed) * BALL_PASS_POWER * 0.5

    def _is_in_penalty_box(self, player):
        """Check if a player is inside their team's penalty box."""
        if player.team == 0:  # Green team (left side)
            return (player.x <= PENALTY_BOX_WIDTH and
                    PENALTY_BOX_Y_TOP <= player.y <= PENALTY_BOX_Y_BOTTOM)
        else:  # Red team (right side)
            return (player.x >= PITCH_WIDTH - PENALTY_BOX_WIDTH and
                    PENALTY_BOX_Y_TOP <= player.y <= PENALTY_BOX_Y_BOTTOM)

    def _execute_set_piece(self):
        """Execute the current set piece (kickoff, throw-in, etc.)."""
        self._clear_ball_context()
        if self.game_state == GameState.KICKOFF:
            self.ball.reset()
            self._place_players_kickoff()
            self._place_players_restart(GameState.KICKOFF, self.kickoff_team, (PITCH_WIDTH/2, PITCH_HEIGHT/2))
            
            team = self.green_team if self.kickoff_team == 0 else self.red_team
            taker = self.np_random.choice(team[1:])  # Randomize outfield taker
            taker.x, taker.y = self.ball.x, self.ball.y
            taker.has_ball = True
            taker.cooldown = 10
            self.ball.owner = taker
            self.ball.last_touch_team = self.kickoff_team

        elif self.game_state == GameState.THROW_IN:
            self.ball.x, self.ball.y = self.set_piece_pos
            # Find closest player of the set piece team
            team = self.green_team if self.set_piece_team == 0 else self.red_team
            closest = min(team[1:], key=lambda p: p.distance_to(self.ball.x, self.ball.y))
            # Teleport nearest player to take the set piece
            closest.x, closest.y = self.ball.x, self.ball.y
            closest.has_ball = True
            closest.cooldown = 10
            self.ball.owner = closest
            self.ball.last_touch_team = self.set_piece_team
            self._place_players_restart(GameState.THROW_IN, self.set_piece_team, self.set_piece_pos)

        elif self.game_state == GameState.GOAL_KICK:
            self.ball.x, self.ball.y = self.set_piece_pos
            # Goalkeeper takes goal kick
            team = self.green_team if self.set_piece_team == 0 else self.red_team
            gk = team[0]
            gk.x, gk.y = self.ball.x, self.ball.y
            self.ball.owner = gk
            gk.has_ball = True
            gk.cooldown = 10
            self.ball.last_touch_team = self.set_piece_team
            self._place_players_restart(GameState.GOAL_KICK, self.set_piece_team, self.set_piece_pos)

        elif self.game_state == GameState.CORNER_KICK:
            self.ball.x, self.ball.y = self.set_piece_pos
            team = self.green_team if self.set_piece_team == 0 else self.red_team
            # Outfield player takes corner
            closest = min(team[1:], key=lambda p: p.distance_to(self.ball.x, self.ball.y))
            closest.x, closest.y = self.ball.x, self.ball.y
            closest.has_ball = True
            closest.cooldown = 10
            self.ball.owner = closest
            self.ball.last_touch_team = self.set_piece_team
            self._place_players_restart(GameState.CORNER_KICK, self.set_piece_team, self.set_piece_pos)

        elif self.game_state == GameState.FREE_KICK:
            self.ball.x, self.ball.y = self.set_piece_pos
            team = self.green_team if self.set_piece_team == 0 else self.red_team
            closest = min(team[1:], key=lambda p: p.distance_to(self.ball.x, self.ball.y))
            closest.x, closest.y = self.ball.x, self.ball.y
            closest.has_ball = True
            closest.cooldown = 10
            self.ball.owner = closest
            self.ball.last_touch_team = self.set_piece_team
            self._place_players_restart(GameState.FREE_KICK, self.set_piece_team, self.set_piece_pos)

    def _update_cooldowns(self):
        """Decrease all player cooldowns."""
        for player in self.all_players:
            if player.cooldown > 0:
                player.cooldown -= 1

    def _update_stamina(self):
        """Update player stamina based on movement."""
        for player in self.all_players:
            speed = math.sqrt(player.vx**2 + player.vy**2)
            if speed > 0.5:
                # Drains slower, min cap at 0.4 (so players don't become snails)
                player.stamina = max(0.4, player.stamina - 0.0003)
            else:
                # Recovers faster
                player.stamina = min(1.0, player.stamina + 0.002)

    def _compute_reward(self, team=0):
        """Compute reward for the specified team (0=green, 1=red)."""
        reward = 0.0

        if team == 0:
            own_score = self.green_score
            prev_own_score = self.prev_green_score
            opp_score = self.red_score
            prev_opp_score = self.prev_red_score
            own_team_id = 0
            opp_team_id = 1
            own_players = self.green_team
            # Positive progress is moving right (increasing x)
            ball_progress = (self.ball.x - self.prev_ball_x) / PITCH_WIDTH
        else:
            own_score = self.red_score
            prev_own_score = self.prev_red_score
            opp_score = self.green_score
            prev_opp_score = self.prev_green_score
            own_team_id = 1
            opp_team_id = 0
            own_players = self.red_team
            # Positive progress is moving left (decreasing x)
            ball_progress = (self.prev_ball_x - self.ball.x) / PITCH_WIDTH

        # Dense shaping
        # small time penalty
        reward -= 0.001
        # small ball progress (team 0 right, team 1 left)
        reward += ball_progress * 0.02

        # Goal and Assist rewards
        if own_score > prev_own_score:
            reward += 5.0
            if self.ball.second_last_touch_team == own_team_id:
                reward += 1.5  # Assist bonus!
        if opp_score > prev_opp_score:
            reward -= 5.0
            if self.ball.second_last_touch_team == opp_team_id:
                reward -= 1.5  # Opponent Assist penalty

        # Win/lose bonus
        if self.done:
            if own_score >= GOALS_TO_WIN:
                reward += 10.0
            elif opp_score >= GOALS_TO_WIN:
                reward -= 10.0
            elif getattr(self, "truncated", False):
                if own_score > opp_score:
                    reward += 10.0
                elif opp_score > own_score:
                    reward -= 10.0

        # Spacing Penalty (Outfield players only to allow GK to stay in goal)
        p1 = own_players[1]
        p2 = own_players[2]
        
        outfield_spacing = math.hypot(p1.x - p2.x, p1.y - p2.y)
        
        if outfield_spacing < 60.0:
            reward -= 0.01

        return reward

    def _get_obs(self, team=0):
        """Get observation for the specified team (0=green, 1=red)."""
        if team == 0:
            own_team = self.green_team
            opp_team = self.red_team
            own_score = self.green_score
            opp_score = self.red_score
        else:
            own_team = self.red_team
            opp_team = self.green_team
            own_score = self.red_score
            opp_score = self.green_score

        obs = np.zeros(18, dtype=np.float32)

        # Own players' positions (normalized)
        for i, p in enumerate(own_team):
            if team == 0:
                obs[i*2] = p.x / PITCH_WIDTH
            else:
                obs[i*2] = (PITCH_WIDTH - p.x) / PITCH_WIDTH
            obs[i*2 + 1] = p.y / PITCH_HEIGHT

        # Opponent players' positions (normalized)
        for i, p in enumerate(opp_team):
            if team == 0:
                obs[6 + i*2] = p.x / PITCH_WIDTH
            else:
                obs[6 + i*2] = (PITCH_WIDTH - p.x) / PITCH_WIDTH
            obs[6 + i*2 + 1] = p.y / PITCH_HEIGHT

        # Ball position and velocity (normalized)
        if team == 0:
            obs[12] = self.ball.x / PITCH_WIDTH
            obs[14] = np.clip(self.ball.vx / BALL_MAX_SPEED, -1, 1) * 0.5 + 0.5
        else:
            obs[12] = (PITCH_WIDTH - self.ball.x) / PITCH_WIDTH
            obs[14] = np.clip((-self.ball.vx) / BALL_MAX_SPEED, -1, 1) * 0.5 + 0.5
            
        obs[13] = self.ball.y / PITCH_HEIGHT
        obs[15] = np.clip(self.ball.vy / BALL_MAX_SPEED, -1, 1) * 0.5 + 0.5

        # Scores (normalized by goals to win)
        obs[16] = own_score / GOALS_TO_WIN
        obs[17] = opp_score / GOALS_TO_WIN

        return obs

    def _get_info(self):
        """Get additional info dict."""
        return {
            "green_score": self.green_score,
            "red_score": self.red_score,
            "steps": self.steps,
            "game_state": self.game_state,
            "ball_pos": (self.ball.x, self.ball.y),
            "ball_owner": (self.ball.owner.team, self.ball.owner.player_id)
                          if self.ball.owner else None,
            "match_stats": self.match_stats,
        }

    def render(self):
        """Render the current state."""
        if self.renderer is None:
            from renderer import FootballRenderer
            self.renderer = FootballRenderer(
                PITCH_WIDTH + PITCH_MARGIN * 2,
                PITCH_HEIGHT + PITCH_MARGIN * 2
            )

        state = {
            "green_team": [(p.x, p.y, p.is_goalkeeper, p.has_ball) for p in self.green_team],
            "red_team": [(p.x, p.y, p.is_goalkeeper, p.has_ball) for p in self.red_team],
            "ball": (self.ball.x, self.ball.y),
            "green_score": self.green_score,
            "red_score": self.red_score,
            "game_state": self.game_state,
            "steps": self.steps,
            "pitch_margin": PITCH_MARGIN,
            "match_stats": self.match_stats,
            "current_episode": getattr(self, "current_episode", 0),
        }

        return self.renderer.render(state)

    def close(self):
        """Clean up renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
