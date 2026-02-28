# ⚽ Football Simulator with Reinforcement Learning

A 3v3 football (soccer) simulator where AI agents learn to play football through **self-play reinforcement learning**. Two teams — **Green** and **Red** — each with 2 outfield players and 1 goalkeeper, compete on a scaled-down pitch. The agents learn to pass, shoot, defend, and save goals using PPO (Proximal Policy Optimization).

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange)
![SB3](https://img.shields.io/badge/Stable--Baselines3-2.2+-red)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to the project directory
cd "Reinforcement learning"

# Install all required packages
pip install -r requirements.txt
```

### 2. Run the Rule Tests (verify everything works)

```bash
python test_rules.py
```

Expected output:
```
✅ Environment creation: PASSED
✅ Kickoff position: PASSED
✅ Goal scoring: PASSED
...
Results: 11 passed, 0 failed out of 11 tests
```

### 3. Train the Agent (with live visualization!)

```bash
# 🎮 Train AND watch the agent play in a Pygame window
python train.py --render --timesteps 2000000

# This trains the agent and **pops up a Pygame window every 20 episodes** so you can watch it improve.
# To resume training later, just add the `--resume` flag:
# python train.py --resume --render --timesteps 2000000

# Headless training (faster, no visuals)
python train.py --timesteps 2000000 --log
# To resume headless training later:
# python train.py --resume --timesteps 2000000 --log
```

### 4. Watch Trained Agent Play

```bash
# Watch 5 matches with the trained model
python play.py --model checkpoints/football_ppo_final --episodes 5

# Watch random agents play (no training needed)
python play.py
```

---

![Gameplay Snapshot](images/playing.png)

## 🏟️ Game Overview

| Feature | Details |
|---------|---------|
| **Teams** | Green (left) vs Red (right) |
| **Players** | 2 outfield + 1 goalkeeper per side |
| **Pitch** | 800×500 px scaled-down pitch |
| **Win condition** | First team to score **2 goals** |
| **Goalkeeper** | Team color jersey with **white stripes** |

### Controls & Actions (AI-controlled)
Each player can perform 2 actions per step:
- **Move**: 8 directions + stay still
- **Kick/Pass**: 8 directions + no kick

---

## 📋 Football Rules Implemented

| Rule | How it works |
|------|-------------|
| **Kickoff** | Ball at center, alternates after each goal |
| **Throw-in** | Ball crosses sideline → opposing team throws in |
| **Goal kick** | Ball crosses end-line (by attacker) → GK kicks |
| **Corner kick** | Ball crosses end-line (by defender) → attacker gets corner |
| **Free kick** | Awarded after a foul, from foul location |
| **Fouls** | Failed tackles may be called as fouls |
| **Offside** | Attacker behind last defender when receiving in opponent's half |
| **GK handling** | Goalkeeper can only hold ball in penalty box |
| **GK movement** | Goalkeeper restricted to own half |
| **Stamina** | Players tire when sprinting, recover when idle |
| **Tackles** | Success depends on distance & stamina |
| **Collisions** | Players physically push each other apart |
| **Ball physics** | Friction, speed caps, realistic movement |

---

## 🤖 RL Architecture

```mermaid
flowchart TD
    subgraph Training Loop
        A[PPO Agent] -->|controls| B[Green Team - learning]
        B --> C[FootballEnv]
        C --> D[Red Team - opponent]
        E[Frozen Copy] -->|controls| D
        A -->|periodic copy| E
        C -->|obs + reward| A
    end
```

- **Algorithm**: PPO with `[256, 256, 128]` networks
- **Self-play**: Opponent updated every 50K steps with frozen copy
- **Observation**: 18D vector (player positions, ball, scores)
- **Actions**: 6 discrete values (3 players × move + kick)

### Reward Signals
| Signal | Value |
|--------|-------|
| Score a goal | +10 |
| Concede a goal | -10 |
| Win match | +20 |
| Lose match | -20 |
| Ball toward goal | +0.05 × progress |
| Possession | +0.005/step (only if also progressing) |
| Shot on target | +1.0 |
| Shot quality (xG) | +2.0 × xG |
| Time penalty | -0.002/step |
| Successful Pass | +0.1 |
| Pass that advances play | +0.2 (extra if forward or breaks a line) |
| Key pass | +0.5 |
| Lose possession | -0.5 |
| Turnover in defensive third | -1.0 |
| Tackle won / interception | +0.3 |
| Block shot / clearance | +0.2 |
| Goalkeeper Save | +2.0 |
| Assist Bonus | +3.0 |

---

## 📁 Project Structure

```
Reinforcement learning/
├── football_env.py       # Core Gymnasium environment + rules
├── renderer.py           # Pygame rendering (pitch, players, ball)
├── self_play_wrapper.py  # Multi-agent → single-agent wrapper
├── train.py              # PPO training with self-play + live viz
├── play.py               # Watch matches with Pygame
├── test_rules.py         # Automated rule verification tests
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── checkpoints/          # Saved model checkpoints (created during training)
└── logs/                 # TensorBoard logs (created with --log flag)
```

---

## ⚙️ Training Options

```bash
python train.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 2,000,000 | Total training steps |
| `--render` | off | Enable live Pygame visualization |
| `--render_every` | 20 | Show match every N episodes |
| `--render_speed` | 1.5 | Playback speed for visual matches |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 256 | Batch size |
| `--selfplay_interval` | 50,000 | Steps between opponent updates |
| `--log` | off | Enable TensorBoard logging |
| `--seed` | 42 | Random seed |

### Watching Training with TensorBoard

```bash
python train.py --timesteps 1000000 --log
tensorboard --logdir logs
# Open http://localhost:6006 in your browser
```

---

## 🎮 Viewer Options

```bash
python play.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | None | Path to trained model (random if not set) |
| `--opponent_model` | None | Path to red team model |
| `--episodes` | 3 | Number of matches |
| `--speed` | 1.0 | Playback speed |
| `--deterministic` | off | Less random action selection |

---

## 📊 Recommended Training Schedule

| Phase | Timesteps | What to Expect |
|-------|-----------|----------------|
| **Early** | 0–200K | Random movement, occasional accidental goals |
| **Mid** | 200K–1M | Basic ball-chasing, some passing attempts |
| **Late** | 1M–2M+ | Coordinated play, passing, goal-saving |

> **Tip**: Run with `--render --render_every 50` to watch improvement over time without slowing training too much.

