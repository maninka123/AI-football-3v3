"""
Football Rules Test Script
===========================
Automated tests to verify that all football rules work correctly.
"""

import numpy as np
import sys

# Add parent directory
sys.path.insert(0, '.')

from football_env import (
    FootballEnv, GameState,
    PITCH_WIDTH, PITCH_HEIGHT,
    GOAL_Y_TOP, GOAL_Y_BOTTOM,
    PENALTY_BOX_WIDTH, PENALTY_BOX_Y_TOP, PENALTY_BOX_Y_BOTTOM,
    BALL_RADIUS
)


def test_environment_creation():
    """Test that environment creates successfully."""
    env = FootballEnv()
    obs, info = env.reset()
    assert obs.shape == (18,), f"Expected obs shape (18,), got {obs.shape}"
    assert 0 <= obs.min() and obs.max() <= 1, "Observations should be normalized [0,1]"
    print("✅ Environment creation: PASSED")
    env.close()


def test_kickoff():
    """Test that kickoff places ball at center and gives possession."""
    env = FootballEnv()
    obs, info = env.reset()

    # Ball should start at center area
    ball_x, ball_y = info["ball_pos"]
    assert abs(ball_x - PITCH_WIDTH / 2) < 50, f"Ball x={ball_x}, expected near center"
    assert abs(ball_y - PITCH_HEIGHT / 2) < 50, f"Ball y={ball_y}, expected near center"
    print("✅ Kickoff position: PASSED")
    env.close()


def test_goal_scoring():
    """Test goal detection."""
    env = FootballEnv()
    env.reset()

    # Force ball into right goal (green scores)
    env.game_state = GameState.PLAYING
    env.ball.owner = None
    env.ball.x = PITCH_WIDTH - BALL_RADIUS  # At the right edge
    env.ball.y = (GOAL_Y_TOP + GOAL_Y_BOTTOM) / 2  # Center of goal
    env.ball.vx = 5
    env.ball.vy = 0
    env.ball.last_touch_team = 0

    prev_green_score = env.green_score
    env._check_goals()

    assert env.green_score == prev_green_score + 1, "Green should have scored"
    assert env.game_state == GameState.KICKOFF, "Should be kickoff after goal"
    print("✅ Goal scoring: PASSED")
    env.close()


def test_sideline_out():
    """Test throw-in when ball goes over sideline."""
    env = FootballEnv()
    env.reset()
    env.game_state = GameState.PLAYING

    # Force ball over top sideline
    env.ball.owner = None
    env.ball.x = 300
    env.ball.y = BALL_RADIUS - 1  # Over the top line
    env.ball.vx = 0
    env.ball.vy = -3
    env.ball.last_touch_team = 0  # Green touched last

    env._check_out_of_bounds()

    assert env.game_state == GameState.THROW_IN, "Should be throw-in"
    assert env.set_piece_team == 1, "Red should get throw-in (green touched last)"
    print("✅ Sideline throw-in: PASSED")
    env.close()


def test_goal_kick():
    """Test goal kick when ball goes over end-line (touched by attacker)."""
    env = FootballEnv()
    env.reset()
    env.game_state = GameState.PLAYING

    # Ball goes over left end-line, last touched by red (attacker)
    env.ball.owner = None
    env.ball.x = BALL_RADIUS - 1
    env.ball.y = 100  # Not in goal area
    env.ball.vx = -3
    env.ball.vy = 0
    env.ball.last_touch_team = 1  # Red touched last

    env._check_out_of_bounds()

    assert env.game_state == GameState.GOAL_KICK, "Should be goal kick"
    assert env.set_piece_team == 0, "Green should get goal kick"
    print("✅ Goal kick: PASSED")
    env.close()


def test_corner_kick():
    """Test corner kick when ball goes over end-line (touched by defender)."""
    env = FootballEnv()
    env.reset()
    env.game_state = GameState.PLAYING

    # Ball goes over left end-line, last touched by green (defender)
    env.ball.owner = None
    env.ball.x = BALL_RADIUS - 1
    env.ball.y = 100  # Not in goal
    env.ball.vx = -3
    env.ball.vy = 0
    env.ball.last_touch_team = 0  # Green touched last (own end-line)

    env._check_out_of_bounds()

    assert env.game_state == GameState.CORNER_KICK, "Should be corner kick"
    assert env.set_piece_team == 1, "Red should get corner"
    print("✅ Corner kick: PASSED")
    env.close()


def test_goalkeeper_restriction():
    """Test that goalkeeper can't handle ball outside penalty box."""
    env = FootballEnv()
    env.reset()
    env.game_state = GameState.PLAYING

    # Move green GK outside penalty box
    gk = env.green_team[0]
    gk.x = PITCH_WIDTH / 2 - 50  # Well outside penalty box
    gk.y = PITCH_HEIGHT / 2
    gk.has_ball = True
    env.ball.owner = gk

    env._enforce_goalkeeper_restrictions()

    assert not gk.has_ball, "GK should lose ball outside penalty box"
    assert env.ball.owner is None, "Ball should be free"
    print("✅ Goalkeeper restriction: PASSED")
    env.close()


def test_win_condition():
    """Test that match ends when a team scores 2 goals."""
    env = FootballEnv()
    obs, info = env.reset()

    # Simulate green scoring 2 goals
    env.green_score = 2
    env.prev_green_score = 1

    # Step to trigger win check
    action = env.action_space.sample()
    env.set_red_actions(np.array([0, 0, 0, 0, 0, 0]))
    obs, reward, done, truncated, info = env.step(action)

    assert done, "Game should be done when green has 2 goals"
    print("✅ Win condition (2 goals): PASSED")
    env.close()


def test_full_episode():
    """Test a full episode runs without crashing."""
    env = FootballEnv()
    obs, info = env.reset()

    for i in range(500):
        action = env.action_space.sample()
        env.set_red_actions(env.action_space.sample())
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()

    print("✅ Full episode (500 steps): PASSED")
    env.close()


def test_penalty_box_detection():
    """Test penalty box boundary detection."""
    env = FootballEnv()
    env.reset()

    # Green GK inside penalty box
    gk = env.green_team[0]
    gk.x = 50
    gk.y = PITCH_HEIGHT / 2
    assert env._is_in_penalty_box(gk), "GK should be in penalty box"

    # Green GK outside penalty box
    gk.x = PENALTY_BOX_WIDTH + 50
    gk.y = PITCH_HEIGHT / 2
    assert not env._is_in_penalty_box(gk), "GK should NOT be in penalty box"

    # Red GK inside penalty box
    red_gk = env.red_team[0]
    red_gk.x = PITCH_WIDTH - 50
    red_gk.y = PITCH_HEIGHT / 2
    assert env._is_in_penalty_box(red_gk), "Red GK should be in penalty box"

    print("✅ Penalty box detection: PASSED")
    env.close()


def test_set_piece_execution():
    """Test that set pieces execute correctly."""
    env = FootballEnv()
    env.reset()

    # Test throw-in execution
    env.game_state = GameState.THROW_IN
    env.set_piece_team = 0
    env.set_piece_pos = (200, 10)
    env._execute_set_piece()

    assert env.ball.owner is not None, "Someone should have the ball after throw-in"
    assert env.ball.owner.team == 0, "Green should have ball after their throw-in"
    print("✅ Set piece execution: PASSED")
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("⚽ Football Simulator — Rule Tests")
    print("=" * 60)

    tests = [
        test_environment_creation,
        test_kickoff,
        test_goal_scoring,
        test_sideline_out,
        test_goal_kick,
        test_corner_kick,
        test_goalkeeper_restriction,
        test_win_condition,
        test_full_episode,
        test_penalty_box_detection,
        test_set_piece_execution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
