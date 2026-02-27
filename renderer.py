"""
Football Simulator Renderer
============================
Pygame-based rendering for the football simulator.
Draws the pitch, players, ball, and scoreboard.
"""

import pygame
import math
import numpy as np

# Colors
PITCH_GREEN = (34, 139, 34)
PITCH_GREEN_LIGHT = (40, 160, 40)
LINE_WHITE = (255, 255, 255)
BALL_WHITE = (255, 255, 255)
BALL_OUTLINE = (50, 50, 50)
GREEN_TEAM_COLOR = (46, 204, 113)
RED_TEAM_COLOR = (231, 76, 60)
GK_STRIPE_COLOR = (255, 255, 255)
GOAL_COLOR = (255, 255, 255)
GOAL_NET_COLOR = (200, 200, 200, 128)
SCOREBOARD_BG = (30, 30, 30)
SCOREBOARD_TEXT = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)
STADIUM_TRACK = (180, 80, 60)
STADIUM_OUTER = (40, 45, 50)

# Import pitch constants
from football_env import (
    PITCH_WIDTH, PITCH_HEIGHT, PITCH_MARGIN,
    GOAL_WIDTH, GOAL_HEIGHT, GOAL_Y_TOP, GOAL_Y_BOTTOM,
    PENALTY_BOX_WIDTH, PENALTY_BOX_HEIGHT,
    PENALTY_BOX_Y_TOP, PENALTY_BOX_Y_BOTTOM,
    GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
    GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM,
    CENTER_CIRCLE_RADIUS,
    PLAYER_RADIUS, BALL_RADIUS,
    GameState
)


class FootballRenderer:
    """Pygame-based football pitch renderer."""

    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("⚽ Football Simulator — Green vs Red")
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 20, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 14)
        except Exception:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)

        self.frame_count = 0

    def render(self, state):
        """Render the current game state."""
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        margin = state.get("pitch_margin", PITCH_MARGIN)

        self.screen.fill(STADIUM_OUTER)
        
        # Draw stadium
        self._draw_stadium(margin)
        
        # Draw grass and lines
        self._draw_pitch(margin)
        self._draw_goals(margin)

        # Draw players
        for px, py, is_gk, has_ball in state["green_team"]:
            self._draw_player(px + margin, py + margin, GREEN_TEAM_COLOR, is_gk, has_ball)

        for px, py, is_gk, has_ball in state["red_team"]:
            self._draw_player(px + margin, py + margin, RED_TEAM_COLOR, is_gk, has_ball)

        # Draw ball
        bx, by = state["ball"]
        self._draw_ball(bx + margin, by + margin)

        # Draw scoreboard
        self._draw_scoreboard(state["green_score"], state["red_score"],
                              state["game_state"], state["steps"], 
                              state.get("current_episode", 0))

        # Draw game state indicator
        self._draw_game_state_indicator(state["game_state"], margin)

        # Draw Match Stats HUD
        if "match_stats" in state:
            self._draw_stats_hud(state["match_stats"], margin)

        pygame.display.flip()
        self.clock.tick(30)
        self.frame_count += 1

        # Return surface as RGB array for gymnasium
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def _draw_stadium(self, margin):
        """Draw the stadium track around the grass pitch."""
        # The running track around the pitch
        track_rect = pygame.Rect(margin - 15, margin - 15, 
                                 PITCH_WIDTH + 30, PITCH_HEIGHT + 30)
        pygame.draw.rect(self.screen, STADIUM_TRACK, track_rect, border_radius=10)
        # Inner white line for track
        pygame.draw.rect(self.screen, (200, 200, 200), track_rect, 2, border_radius=10)
        
        # The actual grass base so lines can be drawn on top
        grass_rect = pygame.Rect(margin, margin, PITCH_WIDTH, PITCH_HEIGHT)
        pygame.draw.rect(self.screen, PITCH_GREEN, grass_rect)

    def _draw_pitch(self, margin):
        """Draw the football pitch with all markings."""
        m = margin

        # Pitch background with alternating grass stripes
        stripe_width = PITCH_WIDTH // 10
        for i in range(10):
            color = PITCH_GREEN if i % 2 == 0 else PITCH_GREEN_LIGHT
            rect = pygame.Rect(m + i * stripe_width, m, stripe_width, PITCH_HEIGHT)
            pygame.draw.rect(self.screen, color, rect)

        # Pitch border (white line)
        pygame.draw.rect(self.screen, LINE_WHITE,
                         (m, m, PITCH_WIDTH, PITCH_HEIGHT), 3)

        # Halfway line
        pygame.draw.line(self.screen, LINE_WHITE,
                         (m + PITCH_WIDTH // 2, m),
                         (m + PITCH_WIDTH // 2, m + PITCH_HEIGHT), 2)

        # Center circle
        pygame.draw.circle(self.screen, LINE_WHITE,
                           (m + PITCH_WIDTH // 2, m + PITCH_HEIGHT // 2),
                           CENTER_CIRCLE_RADIUS, 2)

        # Center spot
        pygame.draw.circle(self.screen, LINE_WHITE,
                           (m + PITCH_WIDTH // 2, m + PITCH_HEIGHT // 2), 4)

        # Left penalty box
        pygame.draw.rect(self.screen, LINE_WHITE,
                         (m, m + int(PENALTY_BOX_Y_TOP),
                          PENALTY_BOX_WIDTH, PENALTY_BOX_HEIGHT), 2)

        # Right penalty box
        pygame.draw.rect(self.screen, LINE_WHITE,
                         (m + PITCH_WIDTH - PENALTY_BOX_WIDTH, m + int(PENALTY_BOX_Y_TOP),
                          PENALTY_BOX_WIDTH, PENALTY_BOX_HEIGHT), 2)

        # Left goal area (6-yard box)
        pygame.draw.rect(self.screen, LINE_WHITE,
                         (m, m + int(GOAL_AREA_Y_TOP),
                          GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT), 2)

        # Right goal area (6-yard box)
        pygame.draw.rect(self.screen, LINE_WHITE,
                         (m + PITCH_WIDTH - GOAL_AREA_WIDTH, m + int(GOAL_AREA_Y_TOP),
                          GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT), 2)

        # Penalty spots
        pygame.draw.circle(self.screen, LINE_WHITE,
                           (m + PENALTY_BOX_WIDTH - 15, m + PITCH_HEIGHT // 2), 3)
        pygame.draw.circle(self.screen, LINE_WHITE,
                           (m + PITCH_WIDTH - PENALTY_BOX_WIDTH + 15, m + PITCH_HEIGHT // 2), 3)

        # Corner arcs
        corner_radius = 15
        for cx, cy in [(m, m), (m, m + PITCH_HEIGHT),
                        (m + PITCH_WIDTH, m), (m + PITCH_WIDTH, m + PITCH_HEIGHT)]:
            pygame.draw.arc(self.screen, LINE_WHITE,
                            (cx - corner_radius, cy - corner_radius,
                             corner_radius * 2, corner_radius * 2),
                            0, math.pi / 2 if cy == m else -math.pi / 2, 2)

    def _draw_goals(self, margin):
        """Draw goal posts and nets."""
        m = margin

        # Left goal (Green team's goal)
        goal_rect_left = pygame.Rect(m - GOAL_WIDTH, m + int(GOAL_Y_TOP),
                                      GOAL_WIDTH, int(GOAL_HEIGHT))
        pygame.draw.rect(self.screen, GOAL_COLOR, goal_rect_left, 3)
        # Goal net (light fill)
        net_surf = pygame.Surface((GOAL_WIDTH, int(GOAL_HEIGHT)), pygame.SRCALPHA)
        net_surf.fill((200, 200, 200, 60))
        self.screen.blit(net_surf, (m - GOAL_WIDTH, m + int(GOAL_Y_TOP)))
        # Net lines
        for i in range(0, int(GOAL_HEIGHT), 10):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (m - GOAL_WIDTH, m + int(GOAL_Y_TOP) + i),
                             (m, m + int(GOAL_Y_TOP) + i), 1)

        # Right goal (Red team's goal)
        goal_rect_right = pygame.Rect(m + PITCH_WIDTH, m + int(GOAL_Y_TOP),
                                       GOAL_WIDTH, int(GOAL_HEIGHT))
        pygame.draw.rect(self.screen, GOAL_COLOR, goal_rect_right, 3)
        net_surf2 = pygame.Surface((GOAL_WIDTH, int(GOAL_HEIGHT)), pygame.SRCALPHA)
        net_surf2.fill((200, 200, 200, 60))
        self.screen.blit(net_surf2, (m + PITCH_WIDTH, m + int(GOAL_Y_TOP)))
        for i in range(0, int(GOAL_HEIGHT), 10):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (m + PITCH_WIDTH, m + int(GOAL_Y_TOP) + i),
                             (m + PITCH_WIDTH + GOAL_WIDTH, m + int(GOAL_Y_TOP) + i), 1)

    def _draw_player(self, x, y, team_color, is_goalkeeper, has_ball):
        """Draw a player as a circle with team color. Goalkeepers get white stripes."""
        x, y = int(x), int(y)
        r = PLAYER_RADIUS

        # Shadow
        pygame.draw.circle(self.screen, (0, 0, 0, 80), (x + 2, y + 3), r)

        # Body circle
        pygame.draw.circle(self.screen, team_color, (x, y), r)

        if is_goalkeeper:
            # White stripes on goalkeeper
            stripe_w = 3
            for offset in [-6, -2, 2, 6]:
                sx = x + offset
                pygame.draw.line(self.screen, GK_STRIPE_COLOR,
                                 (sx, y - r + 3), (sx, y + r - 3), stripe_w)

        # Outline
        outline_color = (255, 255, 100) if has_ball else (20, 20, 20)
        outline_width = 3 if has_ball else 2
        pygame.draw.circle(self.screen, outline_color, (x, y), r, outline_width)

        # Direction indicator (small triangle) — based on the player's movement
        # Just draw a tiny dot in the center for now
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 3)

    def _draw_ball(self, x, y):
        """Draw the football."""
        x, y = int(x), int(y)
        r = BALL_RADIUS

        # Subtle shadow
        pygame.draw.circle(self.screen, (0, 0, 0), (x + 1, y + 2), r + 1)

        # Ball body
        pygame.draw.circle(self.screen, BALL_WHITE, (x, y), r)
        pygame.draw.circle(self.screen, BALL_OUTLINE, (x, y), r, 1)

        # Small pentagon pattern on ball
        pygame.draw.circle(self.screen, (50, 50, 50), (x, y), 2)

    def _draw_scoreboard(self, green_score, red_score, game_state, steps, episode=0):
        """Draw the scoreboard at the top."""
        # Scoreboard background
        sb_width = 300  # slightly wider to fit episode text
        sb_height = 50
        sb_x = (self.width - sb_width) // 2
        sb_y = 2

        # Rounded rectangle
        sb_rect = pygame.Rect(sb_x, sb_y, sb_width, sb_height)
        pygame.draw.rect(self.screen, SCOREBOARD_BG, sb_rect, border_radius=10)
        pygame.draw.rect(self.screen, (100, 100, 100), sb_rect, 2, border_radius=10)

        # Team colors indicators
        pygame.draw.circle(self.screen, GREEN_TEAM_COLOR, (sb_x + 30, sb_y + sb_height // 2), 10)
        pygame.draw.circle(self.screen, RED_TEAM_COLOR, (sb_x + sb_width - 30, sb_y + sb_height // 2), 10)

        # Score text
        score_text = f"{green_score}  -  {red_score}"
        text_surface = self.font_large.render(score_text, True, SCOREBOARD_TEXT)
        text_rect = text_surface.get_rect(center=(sb_x + sb_width // 2, sb_y + sb_height // 2 - 2))
        self.screen.blit(text_surface, text_rect)

        # Team names
        green_label = self.font_small.render("GREEN", True, GREEN_TEAM_COLOR)
        red_label = self.font_small.render("RED", True, RED_TEAM_COLOR)
        self.screen.blit(green_label, (sb_x + 10, sb_y + sb_height + 2))
        self.screen.blit(red_label, (sb_x + sb_width - 35, sb_y + sb_height + 2))

        # Time/steps indicator
        time_text = f"Step: {steps}"
        if episode > 0:
            time_text = f"Ep: {episode} | Step: {steps}"
            
        time_surface = self.font_small.render(time_text, True, (180, 180, 180))
        self.screen.blit(time_surface, (sb_x + sb_width // 2 - 40, sb_y + sb_height + 2))

    def _draw_game_state_indicator(self, game_state, margin):
        """Draw indicator for current game state (kickoff, throw-in, etc.)."""
        state_names = {
            GameState.KICKOFF: "⚽ KICK OFF",
            GameState.THROW_IN: "📥 THROW IN",
            GameState.GOAL_KICK: "🥅 GOAL KICK",
            GameState.CORNER_KICK: "🚩 CORNER KICK",
            GameState.FREE_KICK: "🔴 FREE KICK",
            GameState.GOAL_SCORED: "🎉 GOAL!",
        }

        if game_state in state_names:
            text = state_names[game_state]
            text_surface = self.font_medium.render(text, True, (255, 255, 100))
            text_rect = text_surface.get_rect(
                center=(self.width // 2, self.height - 20)
            )
            # Background
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=5)
            self.screen.blit(text_surface, text_rect)

    def _draw_stats_hud(self, stats, margin):
        """Draw match statistics on the left and right corners of the screen."""
        total_steps = max(stats.get("total_steps", 1), 1)
        green_pos_steps = stats.get("green_possession_steps", 0)
        red_pos_steps = stats.get("red_possession_steps", 0)
        total_pos = max(green_pos_steps + red_pos_steps, 1)

        green_pos = (green_pos_steps / total_pos) * 100
        red_pos = (red_pos_steps / total_pos) * 100

        # Create transparent panels for Green and Red stats
        panel_w, panel_h = 160, 100
        panel_y = margin + 10

        # Green Stats Panel (Left)
        green_panel_x = margin + 10
        pygame.draw.rect(self.screen, (20, 30, 20), (green_panel_x, panel_y, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(self.screen, GREEN_TEAM_COLOR, (green_panel_x, panel_y, panel_w, panel_h), 1, border_radius=8)
        
        self.screen.blit(self.font_small.render("GREEN STATS", True, GREEN_TEAM_COLOR), (green_panel_x + 10, panel_y + 5))
        self.screen.blit(self.font_small.render(f"Possess: {green_pos:.1f}%", True, SCOREBOARD_TEXT), (green_panel_x + 10, panel_y + 25))
        self.screen.blit(self.font_small.render(f"Passes: {stats['green_passes']}", True, SCOREBOARD_TEXT), (green_panel_x + 10, panel_y + 40))
        self.screen.blit(self.font_small.render(f"Shots: {stats['green_shots']}", True, SCOREBOARD_TEXT), (green_panel_x + 10, panel_y + 55))
        self.screen.blit(self.font_small.render(f"Saves: {stats['green_saves']}", True, SCOREBOARD_TEXT), (green_panel_x + 10, panel_y + 70))

        # Red Stats Panel (Right)
        red_panel_x = self.width - margin - panel_w - 10
        pygame.draw.rect(self.screen, (30, 20, 20), (red_panel_x, panel_y, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(self.screen, RED_TEAM_COLOR, (red_panel_x, panel_y, panel_w, panel_h), 1, border_radius=8)
        
        self.screen.blit(self.font_small.render("RED STATS", True, RED_TEAM_COLOR), (red_panel_x + 10, panel_y + 5))
        self.screen.blit(self.font_small.render(f"Possess: {red_pos:.1f}%", True, SCOREBOARD_TEXT), (red_panel_x + 10, panel_y + 25))
        self.screen.blit(self.font_small.render(f"Passes: {stats['red_passes']}", True, SCOREBOARD_TEXT), (red_panel_x + 10, panel_y + 40))
        self.screen.blit(self.font_small.render(f"Shots: {stats['red_shots']}", True, SCOREBOARD_TEXT), (red_panel_x + 10, panel_y + 55))
        self.screen.blit(self.font_small.render(f"Saves: {stats['red_saves']}", True, SCOREBOARD_TEXT), (red_panel_x + 10, panel_y + 70))

    def close(self):
        """Clean up Pygame."""
        pygame.quit()
