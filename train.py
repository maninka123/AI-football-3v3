"""
Football RL Training Script
============================
Trains a PPO agent to play football using self-play.
The agent controls the Green team, and the opponent (Red team)
uses a periodically updated frozen copy of the agent's policy.

Features:
  --render       Opens a Pygame window and shows live matches during training
  --render_every How many training episodes between each visual match (default: 20)

Usage:
    python train.py --render                        # Train with live visuals!
    python train.py --render --render_every 10      # Show game every 10 episodes
    python train.py --timesteps 500000              # Headless training
    python train.py --timesteps 50000 --log         # With TensorBoard logging
"""

import argparse
import os
import time
import copy
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from self_play_wrapper import SelfPlayWrapper

# Path to save/load training state
TRAINING_STATE_FILE = os.path.join("checkpoints", "training_state.json")


class SelfPlayCallback(BaseCallback):
    """
    Callback to update the opponent's policy periodically during training.
    Every `update_interval` steps, the current agent is saved into an opponent pool.
    The environment then samples randomly from this pool to prevent catastrophic forgetting.
    """

    def __init__(self, env, update_interval=50000, pool_size=5, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.update_interval = update_interval
        self.pool_size = pool_size
        self.last_update = 0
        self.n_updates = 0
        self.pool_dir = os.path.join("checkpoints", "opponent_pool")

        # Ensure opponent pool directory exists
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_interval:
            if self.verbose:
                print(f"\n🔄 Self-play update #{self.n_updates + 1} "
                      f"at step {self.num_timesteps}")

            # Save current model into the opponent pool
            model_path = os.path.join(self.pool_dir, f"opponent_{self.n_updates}")
            self.model.save(model_path)

            # Manage pool size (delete oldest if we have too many)
            pool_files = sorted(glob.glob(os.path.join(self.pool_dir, "opponent_*.zip")), 
                                key=os.path.getmtime)
            
            while len(pool_files) > self.pool_size:
                oldest = pool_files.pop(0)
                try:
                    os.remove(oldest)
                except OSError:
                    pass

            # Update the environment with the current pool of opponents
            actual_env = self.env.envs[0]
            try:
                # Load all available opponents from the pool
                opponent_policies = []
                for p_file in pool_files:
                    opponent_policies.append(PPO.load(p_file, env=None))
                
                # We need to add a method `set_opponent_pool` to `SelfPlayWrapper`
                actual_env.set_opponent_pool(opponent_policies)
                if self.verbose:
                    print(f"   ✅ Opponent pool updated ({len(opponent_policies)} models available)")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Failed to update opponent pool: {e}")

            self.last_update = self.num_timesteps
            self.n_updates += 1

        return True


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards and match outcomes."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        # Check for episode end
        infos = self.locals.get("infos", [])
        for info in infos:
            if "green_score" in info and self.locals.get("dones", [False])[0]:
                self.episodes += 1
                green = info["green_score"]
                red = info["red_score"]

                if green >= 2:
                    self.wins += 1
                    result = "🏆 WIN"
                elif red >= 2:
                    self.losses += 1
                    result = "❌ LOSS"
                else:
                    self.draws += 1
                    result = "🤝 DRAW"

                if self.episodes % 50 == 0 and self.verbose:
                    total = max(self.episodes, 1)
                    print(f"\n📊 Episode {self.episodes}: {result} "
                          f"(Score: {green}-{red})")
                    print(f"   Win rate: {self.wins/total*100:.1f}% | "
                          f"Wins: {self.wins} | Losses: {self.losses} | "
                          f"Draws: {self.draws}")

        return True


class LivePlotCallback(BaseCallback):
    """
    Real-time matplotlib plots that update during training.
    Opens a window with 3 subplots:
      1. Episode Reward (with rolling average) — shows convergence
      2. Win Rate % over time — shows learning progress
      3. Goals Scored vs Conceded — shows offensive/defensive ability
    """

    def __init__(self, plot_every=5, rolling_window=50, verbose=1):
        super().__init__(verbose)
        self.plot_every = plot_every  # Update plots every N episodes
        self.rolling_window = rolling_window
        self.episodes = 0

        # Data storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.green_goals_history = []
        self.red_goals_history = []
        self.cumulative_wins = 0
        self.cumulative_losses = 0
        self.cumulative_draws = 0

        # Current episode tracking
        self._current_reward = 0.0

        # Plot setup
        self.fig = None
        self.axes = None
        self._initialized = False

    def _init_plot(self):
        """Initialize the matplotlib figure with 3 subplots."""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 9))
        self.fig.suptitle('Football RL Training — Live Dashboard',
                          fontsize=14, fontweight='bold', y=0.98)
        self.fig.set_facecolor('#f4f6f9')

        colors = {
            'bg': '#f4f6f9',
            'panel': '#ffffff',
            'green': '#27ae60',
            'red': '#c0392b',
            'yellow': '#e67e22',
            'blue': '#2980b9',
            'text': '#2c3e50',
            'gray': '#7f8c8d',
            'grid': '#ecf0f1',
        }
        self.colors = colors

        for ax in self.axes:
            ax.set_facecolor(colors['panel'])
            ax.tick_params(colors=colors['gray'], labelcolor=colors['gray'])
            ax.spines['bottom'].set_color(colors['gray'])
            ax.spines['left'].set_color(colors['gray'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.8, color=colors['grid'])

        # Subplot 1: Episode Reward
        self.axes[0].set_title('Episode Reward (Convergence)', color=colors['text'],
                               fontsize=11, fontweight='bold', pad=8)
        self.axes[0].set_xlabel('Episode', color=colors['gray'], fontsize=9)
        self.axes[0].set_ylabel('Reward', color=colors['gray'], fontsize=9)

        # Subplot 2: Win Rate
        self.axes[1].set_title('Win Rate Over Time', color=colors['text'],
                               fontsize=11, fontweight='bold', pad=8)
        self.axes[1].set_xlabel('Episode', color=colors['gray'], fontsize=9)
        self.axes[1].set_ylabel('Win Rate %', color=colors['gray'], fontsize=9)
        self.axes[1].set_ylim(-5, 105)

        # Subplot 3: Goals
        self.axes[2].set_title('Goals Scored vs Conceded', color=colors['text'],
                               fontsize=11, fontweight='bold', pad=8)
        self.axes[2].set_xlabel('Episode', color=colors['gray'], fontsize=9)
        self.axes[2].set_ylabel('Goals', color=colors['gray'], fontsize=9)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)
        self._initialized = True

    def _on_step(self) -> bool:
        # Accumulate reward
        rewards = self.locals.get('rewards', [0])
        self._current_reward += rewards[0]

        # Check for episode end
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [False])

        for i, info in enumerate(infos):
            if dones[i] and 'green_score' in info:
                self.episodes += 1
                green = info['green_score']
                red = info['red_score']

                # Store data
                self.episode_rewards.append(self._current_reward)
                self.green_goals_history.append(green)
                self.red_goals_history.append(red)

                # Track wins
                if green >= 2:
                    self.cumulative_wins += 1
                elif red >= 2:
                    self.cumulative_losses += 1
                else:
                    self.cumulative_draws += 1

                total = self.cumulative_wins + self.cumulative_losses + self.cumulative_draws
                win_rate = (self.cumulative_wins / total * 100) if total > 0 else 0
                self.win_rates.append(win_rate)

                # Reset episode reward
                self._current_reward = 0.0

                # Update plots periodically
                if self.episodes % self.plot_every == 0:
                    self._update_plots()

        return True

    def _update_plots(self):
        """Redraw all 3 plots with latest data."""
        if not self._initialized:
            self._init_plot()

        c = self.colors
        episodes_x = list(range(1, len(self.episode_rewards) + 1))

        # --- Plot 1: Episode Reward ---
        ax1 = self.axes[0]
        ax1.clear()
        ax1.set_facecolor(c['panel'])
        ax1.grid(True, alpha=0.8, color=c['grid'])
        ax1.set_title('Episode Reward (Convergence)', color=c['text'],
                       fontsize=11, fontweight='bold', pad=8)

        # Raw rewards (translucent)
        ax1.plot(episodes_x, self.episode_rewards,
                 color=c['blue'], alpha=0.35, linewidth=1.0, label='Raw')

        # Rolling average
        if len(self.episode_rewards) >= self.rolling_window:
            rolling = np.convolve(self.episode_rewards,
                                  np.ones(self.rolling_window) / self.rolling_window,
                                  mode='valid')
            rolling_x = list(range(self.rolling_window, len(self.episode_rewards) + 1))
            ax1.plot(rolling_x, rolling, color=c['yellow'],
                     linewidth=2.5, label=f'Rolling Avg ({self.rolling_window})')
        elif len(self.episode_rewards) >= 5:
            w = min(len(self.episode_rewards), 10)
            rolling = np.convolve(self.episode_rewards,
                                  np.ones(w) / w, mode='valid')
            rolling_x = list(range(w, len(self.episode_rewards) + 1))
            ax1.plot(rolling_x, rolling, color=c['yellow'],
                     linewidth=2.5, label=f'Rolling Avg ({w})')

        ax1.legend(loc='upper left', fontsize=8, facecolor=c['panel'],
                   edgecolor=c['grid'], labelcolor=c['text'])
        ax1.set_xlabel('Episode', color=c['gray'], fontsize=9)
        ax1.set_ylabel('Reward', color=c['gray'], fontsize=9)
        ax1.tick_params(colors=c['gray'], labelcolor=c['gray'])

        # --- Plot 2: Win Rate ---
        ax2 = self.axes[1]
        ax2.clear()
        ax2.set_facecolor(c['panel'])
        ax2.grid(True, alpha=0.8, color=c['grid'])
        ax2.set_title('Win Rate Over Time', color=c['text'],
                       fontsize=11, fontweight='bold', pad=8)

        ax2.fill_between(episodes_x, self.win_rates, alpha=0.3, color=c['green'])
        ax2.plot(episodes_x, self.win_rates, color=c['green'],
                 linewidth=2, label='Win Rate')

        # 50% reference line
        ax2.axhline(y=50, color=c['gray'], linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(len(episodes_x) * 0.02, 52, '50%', color=c['gray'],
                 fontsize=8, alpha=0.7)

        # Current stats annotation
        total = self.cumulative_wins + self.cumulative_losses + self.cumulative_draws
        stats_text = (f'W:{self.cumulative_wins}  L:{self.cumulative_losses}  '
                      f'D:{self.cumulative_draws}')
        ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
                 fontsize=9, color=c['text'], ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=c['bg'], edgecolor=c['grid'], alpha=0.9))

        ax2.set_ylim(-5, 105)
        ax2.set_xlabel('Episode', color=c['gray'], fontsize=9)
        ax2.set_ylabel('Win Rate %', color=c['gray'], fontsize=9)
        ax2.tick_params(colors=c['gray'], labelcolor=c['gray'])

        # --- Plot 3: Goals ---
        ax3 = self.axes[2]
        ax3.clear()
        ax3.set_facecolor(c['panel'])
        ax3.grid(True, alpha=0.8, color=c['grid'])
        ax3.set_title('Goals Scored vs Conceded', color=c['text'],
                       fontsize=11, fontweight='bold', pad=8)

        # Rolling goals average
        w = min(max(len(self.green_goals_history) // 10, 5), 50)
        if len(self.green_goals_history) >= w:
            green_rolling = np.convolve(self.green_goals_history,
                                         np.ones(w) / w, mode='valid')
            red_rolling = np.convolve(self.red_goals_history,
                                       np.ones(w) / w, mode='valid')
            roll_x = list(range(w, len(self.green_goals_history) + 1))
            ax3.plot(roll_x, green_rolling, color=c['green'],
                     linewidth=2.5, label='Green (scored)', zorder=3)
            ax3.plot(roll_x, red_rolling, color=c['red'],
                     linewidth=2.5, label='Red (conceded)', zorder=3)
            ax3.fill_between(roll_x, green_rolling, red_rolling,
                             where=[g > r for g, r in zip(green_rolling, red_rolling)],
                             alpha=0.15, color=c['green'], interpolate=True)
            ax3.fill_between(roll_x, green_rolling, red_rolling,
                             where=[g <= r for g, r in zip(green_rolling, red_rolling)],
                             alpha=0.15, color=c['red'], interpolate=True)
        else:
            ax3.bar(episodes_x, self.green_goals_history,
                    color=c['green'], alpha=0.6, width=0.8, label='Green (scored)')
            ax3.bar(episodes_x, self.red_goals_history,
                    color=c['red'], alpha=0.4, width=0.4, label='Red (conceded)')

        ax3.legend(loc='upper left', fontsize=8, facecolor=c['panel'],
                   edgecolor=c['grid'], labelcolor=c['text'])
        ax3.set_xlabel('Episode', color=c['gray'], fontsize=9)
        ax3.set_ylabel('Avg Goals', color=c['gray'], fontsize=9)
        ax3.tick_params(colors=c['gray'], labelcolor=c['gray'])

        # Refresh
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass

    def _on_training_end(self):
        """Save final plot and keep window open."""
        self._update_plots()
        try:
            self.fig.savefig('training_plots.png', dpi=150,
                             facecolor=self.colors['bg'],
                             bbox_inches='tight')
            if self.verbose:
                print(f"\n📈 Training plots saved to: training_plots.png")
        except Exception:
            pass
        plt.ioff()
        plt.show(block=False)


class LiveVisualizationCallback(BaseCallback):
    """
    Periodically pauses training to play a full visual match in a Pygame window.
    This lets you WATCH the agent learn in real-time while training continues.

    Every `render_every` training episodes, a separate rendered environment is
    created and one full match is played using the current policy, so you can
    see how the agent is improving.
    """

    def __init__(self, render_every=20, speed=1.5, verbose=1):
        super().__init__(verbose)
        self.render_every = render_every
        self.speed = speed
        self.episodes_seen = 0
        self.vis_env = None  # Created lazily

    def _on_step(self) -> bool:
        # Count episodes
        infos = self.locals.get("infos", [])
        for info in infos:
            if "green_score" in info and self.locals.get("dones", [False])[0]:
                self.episodes_seen += 1

                if self.episodes_seen % self.render_every == 0:
                    self._play_visual_match()

        return True

    def _play_visual_match(self):
        """Play one full match with Pygame rendering using the current policy."""
        import pygame

        if self.verbose:
            print(f"\n🎮 Playing visual match #{self.episodes_seen // self.render_every}  "
                  f"(after {self.episodes_seen} training episodes, "
                  f"{self.num_timesteps:,} steps)")

        # Create a rendered environment (separate from training env)
        if self.vis_env is None:
            self.vis_env = SelfPlayWrapper(render_mode="human")

        # Copy the current opponent policy from the training env
        training_env = self.model.get_env().envs[0]
        if hasattr(training_env, 'opponent_policy'):
            self.vis_env.set_opponent_policy(training_env.opponent_policy)

        # Pass the current episode down to the environment
        obs, info = self.vis_env.reset(options={"episode": self.episodes_seen})
        done = False
        total_reward = 0
        match_steps = 0

        while not done:
            # Use the CURRENT agent policy (being trained) for green team
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.vis_env.step(action)
            total_reward += reward
            match_steps += 1
            done = done or truncated

            # Handle Pygame events (prevent "not responding")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.verbose:
                        print("   ⏭️  Visual match skipped by user")
                    self.vis_env.close()
                    self.vis_env = None
                    return

            # Control playback speed
            time.sleep(max(0, (1.0 / 30) / self.speed))

        green_score = info.get("green_score", 0)
        red_score = info.get("red_score", 0)
        if green_score > red_score:
            result = "🟢 GREEN WINS!"
        elif red_score > green_score:
            result = "🔴 RED WINS!"
        else:
            result = "🤝 DRAW"

        if self.verbose:
            print(f"   {result} | Score: {green_score}-{red_score} | "
                  f"Steps: {match_steps} | Reward: {total_reward:.1f}")
            print(f"   ▶️  Training continues...\n")

    def _on_training_end(self):
        """Clean up visualization environment."""
        if self.vis_env is not None:
            self.vis_env.close()
            self.vis_env = None


def make_env(render_mode=None):
    """Create a wrapped football environment."""
    def _init():
        env = SelfPlayWrapper(render_mode=render_mode)
        return env
    return _init


def find_latest_checkpoint():
    """Find the latest checkpoint file in the checkpoints directory."""
    # Look for the final model first
    final_path = os.path.join("checkpoints", "football_ppo_final.zip")
    if os.path.exists(final_path):
        return final_path

    # Find numbered checkpoints and pick the latest
    pattern = os.path.join("checkpoints", "football_ppo_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    # Extract step numbers and find max
    def get_steps(path):
        basename = os.path.basename(path)
        try:
            parts = basename.replace(".zip", "").split("_")
            return int(parts[-2])  # e.g. football_ppo_50000_steps.zip
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=get_steps)
    return checkpoints[-1]


def save_training_state(timesteps_done, stats):
    """Save training progress to JSON so we can resume later."""
    state = {
        "timesteps_done": timesteps_done,
        "episodes": stats.get("episodes", 0),
        "wins": stats.get("wins", 0),
        "losses": stats.get("losses", 0),
        "draws": stats.get("draws", 0),
    }
    os.makedirs("checkpoints", exist_ok=True)
    with open(TRAINING_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return state


def load_training_state():
    """Load previous training progress."""
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE, "r") as f:
            return json.load(f)
    return None


def train(args):
    """Main training function with resume support."""
    print("=" * 60)
    print("⚽ Football Simulator — Reinforcement Learning Training")
    print("=" * 60)

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ---- RESUME LOGIC ----
    resumed = False
    prev_timesteps = 0
    prev_state = None

    if args.resume:
        checkpoint_path = find_latest_checkpoint()
        prev_state = load_training_state()

        if checkpoint_path and prev_state:
            prev_timesteps = prev_state["timesteps_done"]
            remaining = args.timesteps - prev_timesteps

            if remaining <= 0:
                print(f"\n⚠️  Already trained {prev_timesteps:,} steps, "
                      f"which meets your target of {args.timesteps:,}.")
                print(f"   To train more, use a higher --timesteps value.")
                print(f"   Example: python train.py --resume --timesteps {prev_timesteps + 500000}")
                return None

            print(f"\n🔄 RESUMING from checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"   Previous progress: {prev_timesteps:,} steps done")
            print(f"   Previous stats:    W:{prev_state['wins']} L:{prev_state['losses']} D:{prev_state['draws']}")
            print(f"   Target total:      {args.timesteps:,} steps")
            print(f"   Remaining:         {remaining:,} steps")
            resumed = True
        elif checkpoint_path:
            print(f"\n🔄 Found checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"   (No training state file — starting fresh from this model)")
            resumed = True
        else:
            print(f"\n⚠️  No checkpoint found. Starting fresh training.")
            args.resume = False

    # Print config
    print(f"\n  Total timesteps:    {args.timesteps:,}")
    if resumed:
        print(f"  Already completed:  {prev_timesteps:,}")
        print(f"  Remaining:          {args.timesteps - prev_timesteps:,}")
    print(f"  Self-play update:   every {args.selfplay_interval:,} steps")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Live visualization: {'Every ' + str(args.render_every) + ' episodes' if args.render else 'Off'}")
    print(f"  Live plots:         {'On (3 charts)' if args.plots else 'Off'}")
    print(f"  TensorBoard log:    {'Yes' if args.log else 'No'}")
    print("=" * 60)

    if args.render:
        print("\n🎮 Live visualization is ON! A Pygame window will open")
        print(f"   and show a match every {args.render_every} training episodes.")
        print("   You can watch the agent learn in real-time!\n")

    # Create vectorized environment (training env is always headless for speed)
    env = DummyVecEnv([make_env()])

    # ---- LOAD OR CREATE MODEL ----
    if resumed and checkpoint_path:
        print(f"\n📂 Loading model from: {os.path.basename(checkpoint_path)}")
        model = PPO.load(
            checkpoint_path.replace(".zip", ""),
            env=env,
            tensorboard_log="logs" if args.log else None,
        )
        # Update learning rate if changed
        model.learning_rate = args.lr
        print(f"   ✅ Model loaded successfully!")
    else:
        # Create fresh PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
            ),
            tensorboard_log="logs" if args.log else None,
            verbose=1,
            seed=args.seed,
        )

    print(f"\n🏗️  Model architecture:")
    print(f"   Policy network:  [256, 256, 128]")
    print(f"   Value network:   [256, 256, 128]")
    print(f"   Observation dim: {env.observation_space.shape}")
    print(f"   Action dim:      {env.action_space.shape}\n")

    # Callbacks
    reward_callback = RewardLoggerCallback(verbose=1)
    # Restore previous stats if resuming
    if prev_state:
        reward_callback.wins = prev_state.get("wins", 0)
        reward_callback.losses = prev_state.get("losses", 0)
        reward_callback.draws = prev_state.get("draws", 0)
        reward_callback.episodes = prev_state.get("episodes", 0)

    callbacks = [
        SelfPlayCallback(env, update_interval=args.selfplay_interval),
        reward_callback,
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path="checkpoints",
            name_prefix="football_ppo"
        ),
    ]

    # Add live plots callback
    if args.plots:
        callbacks.append(
            LivePlotCallback(
                plot_every=args.plot_every,
                rolling_window=50,
                verbose=1,
            )
        )

    # Add live visualization callback if --render is enabled
    if args.render:
        callbacks.append(
            LiveVisualizationCallback(
                render_every=args.render_every,
                speed=args.render_speed,
                verbose=1,
            )
        )

    # Calculate remaining timesteps
    train_timesteps = args.timesteps - prev_timesteps

    # Train!
    print(f"🚀 {'Resuming' if resumed else 'Starting'} training "
          f"({train_timesteps:,} steps)...\n")
    try:
        model.learn(
            total_timesteps=train_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not resumed,  # Don't reset step counter if resuming
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user.")
        print("   Saving checkpoint so you can resume later...")

    # Save final model
    final_path = os.path.join("checkpoints", "football_ppo_final")
    model.save(final_path)
    print(f"\n💾 Final model saved to: {final_path}")

    # Save training state for resume
    total_timesteps_done = prev_timesteps + model.num_timesteps
    save_training_state(total_timesteps_done, {
        "episodes": reward_callback.episodes,
        "wins": reward_callback.wins,
        "losses": reward_callback.losses,
        "draws": reward_callback.draws,
    })
    print(f"📋 Training state saved ({total_timesteps_done:,} total steps)")
    print(f"   Resume anytime with: python train.py --resume --timesteps <TARGET>")

    # Print final stats
    total = max(reward_callback.episodes, 1)
    print(f"\n{'=' * 60}")
    print(f"📊 Training {'Complete' if model.num_timesteps >= train_timesteps else 'Paused'}!")
    print(f"   Total steps: {total_timesteps_done:,}")
    print(f"   Episodes:   {reward_callback.episodes}")
    print(f"   Wins:       {reward_callback.wins} ({reward_callback.wins/total*100:.1f}%)")
    print(f"   Losses:     {reward_callback.losses} ({reward_callback.losses/total*100:.1f}%)")
    print(f"   Draws:      {reward_callback.draws} ({reward_callback.draws/total*100:.1f}%)")
    print(f"{'=' * 60}")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Football RL Agent")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps (default: 2M)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per rollout (default: 2048)")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--selfplay_interval", type=int, default=50000,
                        help="Opponent update interval (default: 50K)")
    parser.add_argument("--checkpoint_freq", type=int, default=50000,
                        help="Checkpoint frequency (default: 50K)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--log", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--render", action="store_true",
                        help="Enable live visualization during training")
    parser.add_argument("--render_every", type=int, default=20,
                        help="Show a visual match every N training episodes (default: 20)")
    parser.add_argument("--render_speed", type=float, default=1.5,
                        help="Playback speed for visual matches (default: 1.5x)")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Enable live training plots (default: on)")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Disable live training plots")
    parser.add_argument("--plot_every", type=int, default=5,
                        help="Update plots every N episodes (default: 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    train(args)
