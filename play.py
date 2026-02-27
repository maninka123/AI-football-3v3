"""
Football Simulator — Game Viewer / Demo
========================================
Watch trained agents (or random agents) play football.

Usage:
    python play.py                          # Watch random agents play
    python play.py --model checkpoints/football_ppo_final  # Watch trained agent
    python play.py --episodes 5             # Watch 5 matches
    python play.py --speed 2                # 2x speed
"""

import argparse
import time
import sys
import numpy as np
from stable_baselines3 import PPO
from self_play_wrapper import SelfPlayWrapper


def play(args):
    """Run the football simulator with visualization."""
    print("=" * 60)
    print("⚽ Football Simulator — Game Viewer")
    print("=" * 60)

    # Load model if provided
    green_model = None
    red_model = None

    if args.model:
        print(f"📂 Loading green team model: {args.model}")
        try:
            green_model = PPO.load(args.model)
            print("   ✅ Green team model loaded!")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   Using random actions instead.")

    if args.opponent_model:
        print(f"📂 Loading red team model: {args.opponent_model}")
        try:
            red_model = PPO.load(args.opponent_model)
            print("   ✅ Red team model loaded!")
        except Exception as e:
            print(f"   ❌ Failed to load opponent model: {e}")
            print("   Using random actions instead.")

    # Create environment with rendering
    env = SelfPlayWrapper(render_mode="human", opponent_policy=red_model)

    print(f"\n🏟️  Starting {args.episodes} match(es)...")
    print("   Press Ctrl+C to stop.\n")

    total_green_wins = 0
    total_red_wins = 0
    total_draws = 0

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            print(f"--- Match {episode + 1} ---")

            while not done:
                # Get green team action
                if green_model is not None:
                    action, _ = green_model.predict(obs, deterministic=args.deterministic)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                done = done or truncated

                # Control speed
                if args.speed < 3:
                    time.sleep(max(0, (1.0 / 30) / args.speed))

            # Match result
            green_score = info.get("green_score", 0)
            red_score = info.get("red_score", 0)

            if green_score > red_score:
                result = "🟢 GREEN WINS!"
                total_green_wins += 1
            elif red_score > green_score:
                result = "🔴 RED WINS!"
                total_red_wins += 1
            else:
                result = "🤝 DRAW"
                total_draws += 1

            print(f"   Result: {result} | Score: {green_score}-{red_score} | "
                  f"Steps: {step_count} | Reward: {total_reward:.1f}")

            # Brief pause between matches
            if episode < args.episodes - 1:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⏸️  Stopped by user.")

    # Summary
    total = total_green_wins + total_red_wins + total_draws
    if total > 0:
        print(f"\n{'=' * 60}")
        print(f"📊 Session Summary ({total} matches)")
        print(f"   🟢 Green wins: {total_green_wins} ({total_green_wins/total*100:.0f}%)")
        print(f"   🔴 Red wins:   {total_red_wins} ({total_red_wins/total*100:.0f}%)")
        print(f"   🤝 Draws:      {total_draws} ({total_draws/total*100:.0f}%)")
        print(f"{'=' * 60}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Simulator Viewer")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained green team model")
    parser.add_argument("--opponent_model", type=str, default=None,
                        help="Path to trained red team model")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of matches to play (default: 3)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (less random)")
    args = parser.parse_args()

    play(args)
