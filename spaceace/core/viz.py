"""Real-time visual renderer for SpaceAce using pygame."""

import math
import time
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pygame

# Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 65)
DARK_GREEN = (0, 204, 51)
LIGHT_GREEN = (101, 255, 122)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GRAY = (128, 128, 128)


def extract_game_info(obs, info, env):
    """Build visualization data from a raw 19-dim observation and a SpaceAceDirectEnv."""
    detailed_info = env.get_info()

    try:
        geom = env.get_map_geometry()
        map_lines = [(l[0], l[1], l[2], l[3]) for l in geom.get("map_lines", []) if len(l) >= 4]
        pickups = [{"x": p[0], "y": p[1], "collected": p[2]} for p in geom.get("pickup_positions", []) if len(p) >= 3]
        b = geom.get("bounds", {})
        map_bounds = {
            "min_x": b.get("min_x", 0), "max_x": b.get("max_x", 1200),
            "min_y": b.get("min_y", 0), "max_y": b.get("max_y", 800),
            "lines": map_lines,
        }
    except Exception:
        map_bounds = {"min_x": 0, "max_x": 1200, "min_y": 0, "max_y": 800, "lines": []}
        pickups = []

    game_state = {
        "ship_x": float(obs[0]),
        "ship_y": float(obs[1]),
        "ship_vx": float(obs[2]),
        "ship_vy": float(obs[3]),
        "ship_rotation": float(obs[4]),
        "pickups_remaining": int(obs[16]),
        "game_time": detailed_info.get("step_count", 0) / 60.0,
        "ship_exploded": detailed_info.get("ship_exploded", False),
        "level_completed": detailed_info.get("level_completed", False),
    }
    return game_state, pickups, map_bounds


class VisualRenderer:
    """Real-time pygame renderer for SpaceAce."""

    def __init__(self, width=1200, height=800, scale=0.8):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SpaceAce RL Agent - Real-time Visualization")

        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        self.particles = []
        self.camera_x = 0
        self.camera_y = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Debug recording (toggle with X key)
        self.recording = False
        self.record_log = []

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        screen_x = int((world_x - self.camera_x) * self.scale + self.width * 0.5)
        screen_y = int((world_y - self.camera_y) * self.scale + self.height * 0.5)
        return screen_x, screen_y

    def update_camera(self, ship_x: float, ship_y: float):
        self.camera_x = ship_x
        self.camera_y = ship_y

    def draw_grid(self):
        grid_size = 50
        start_x = int(-self.camera_x * self.scale) % grid_size
        start_y = int(-self.camera_y * self.scale) % grid_size
        for x in range(start_x, self.width, grid_size):
            pygame.draw.line(self.screen, (0, 40, 0), (x, 0), (x, self.height), 1)
        for y in range(start_y, self.height, grid_size):
            pygame.draw.line(self.screen, (0, 40, 0), (0, y), (self.width, y), 1)

    def draw_boundaries(self, map_bounds):
        top_left = self.world_to_screen(map_bounds['min_x'], map_bounds['min_y'])
        top_right = self.world_to_screen(map_bounds['max_x'], map_bounds['min_y'])
        bottom_right = self.world_to_screen(map_bounds['max_x'], map_bounds['max_y'])
        bottom_left = self.world_to_screen(map_bounds['min_x'], map_bounds['max_y'])
        points = [top_left, top_right, bottom_right, bottom_left]
        pygame.draw.lines(self.screen, GREEN, True, points, 3)
        pygame.draw.lines(self.screen, DARK_GREEN, True, points, 1)

    def draw_map_lines(self, map_bounds):
        if 'lines' not in map_bounds or not map_bounds['lines']:
            return
        for line in map_bounds['lines']:
            x1, y1, x2, y2 = line
            screen_x1, screen_y1 = self.world_to_screen(x1, y1)
            screen_x2, screen_y2 = self.world_to_screen(x2, y2)
            if (min(screen_x1, screen_x2) > self.width + 100 or
                max(screen_x1, screen_x2) < -100 or
                min(screen_y1, screen_y2) > self.height + 100 or
                max(screen_y1, screen_y2) < -100):
                continue
            pygame.draw.line(self.screen, GREEN, (screen_x1, screen_y1), (screen_x2, screen_y2), 2)
            pygame.draw.line(self.screen, DARK_GREEN, (screen_x1, screen_y1), (screen_x2, screen_y2), 1)

    def draw_ship(self, ship_x, ship_y, ship_rotation, ship_vx, ship_vy, thrusting, exploded):
        screen_x, screen_y = self.world_to_screen(ship_x, ship_y)

        if exploded:
            for i in range(10):
                angle = (i / 10) * 2 * math.pi
                exp_x = screen_x + math.cos(angle) * (20 + np.random.random() * 10)
                exp_y = screen_y + math.sin(angle) * (20 + np.random.random() * 10)
                pygame.draw.circle(self.screen, RED, (int(exp_x), int(exp_y)), 3)
            return

        ship_verts = [
            (0.0, -36.5), (-19.0, 23.5), (-24.0, 23.5), (-15.675, 13.0),
            (19.0, 23.5), (24.0, 23.5), (15.675, 13.0), (0.0, 67.45),
            (-14.1075, 13.0), (14.1075, 13.0),
        ]

        cos_r = math.cos(ship_rotation)
        sin_r = math.sin(ship_rotation)

        transformed_verts = []
        for vx, vy in ship_verts:
            rx = vx * cos_r - vy * sin_r + ship_x
            ry = vx * sin_r + vy * cos_r + ship_y
            transformed_verts.append(self.world_to_screen(rx, ry))

        ship_color = GREEN
        line_width = 3
        pygame.draw.line(self.screen, ship_color, transformed_verts[3], transformed_verts[6], line_width)
        body_points = [transformed_verts[i] for i in [2, 1, 0, 4, 5]]
        pygame.draw.lines(self.screen, ship_color, False, body_points, line_width)

        if thrusting:
            thrust_points = [transformed_verts[i] for i in [8, 7, 9]]
            pygame.draw.lines(self.screen, GREEN, False, thrust_points, 2)

        vel_magnitude = math.sqrt(ship_vx**2 + ship_vy**2)
        if vel_magnitude > 1:
            vel_end_x = screen_x + ship_vx * 2
            vel_end_y = screen_y + ship_vy * 2
            pygame.draw.line(self.screen, MAGENTA, (screen_x, screen_y),
                             (int(vel_end_x), int(vel_end_y)), 2)
            vel_text = self.font_small.render(f"V:{vel_magnitude:.1f}", True, MAGENTA)
            self.screen.blit(vel_text, (screen_x + 20, screen_y - 10))

    def draw_pickups(self, pickups):
        for pickup in pickups:
            if not pickup['collected']:
                screen_x, screen_y = self.world_to_screen(pickup['x'], pickup['y'])
                pulse = (math.sin(self.frame_count * 0.1) + 1) * 0.3 + 0.4
                outer_radius = int(15 * pulse)
                pygame.draw.circle(self.screen, (0, 255, 0, 50), (screen_x, screen_y), outer_radius)
                inner_radius = int(8 * pulse)
                pygame.draw.circle(self.screen, CYAN, (screen_x, screen_y), inner_radius)
                for i in range(4):
                    angle = (self.frame_count * 0.05 + i * math.pi/2) % (2 * math.pi)
                    spark_x = screen_x + math.cos(angle) * 20
                    spark_y = screen_y + math.sin(angle) * 20
                    pygame.draw.circle(self.screen, WHITE, (int(spark_x), int(spark_y)), 2)

    def draw_hud(self, game_state, info, reward, action=None):
        hud_x = 10
        hud_y = 10
        line_height = 25

        stats = [
            f"SpaceAce RL Agent - Level {game_state.get('level', 1)}",
            f"Time: {game_state.get('game_time', 0):.1f}s",
            f"Position: ({game_state['ship_x']:.0f}, {game_state['ship_y']:.0f})",
            f"Velocity: ({game_state['ship_vx']:.1f}, {game_state['ship_vy']:.1f})",
            f"Rotation: {game_state['ship_rotation']:.2f} rad ({math.degrees(game_state['ship_rotation']):.0f})",
            f"Pickups: {game_state['pickups_remaining']}",
            f"Reward: {reward:.2f}",
            f"Step: {info.get('step_count', 0)}",
        ]

        if action is not None:
            action_names = ["NONE", "THRUST", "LEFT", "LEFT+THRUST", "RIGHT", "RIGHT+THRUST", "LEFT+RIGHT", "ALL"]
            action_idx = int(action[0]) * 4 + int(action[1]) * 2 + int(action[2]) if len(action) == 3 else 0
            stats.append(f"Action: {action_names[action_idx]}")

        if game_state.get('ship_exploded', False):
            stats.append("Status: CRASHED")
            status_color = RED
        elif game_state.get('level_completed', False):
            stats.append("Status: COMPLETED")
            status_color = GREEN
        else:
            stats.append("Status: ACTIVE")
            status_color = WHITE

        panel_height = len(stats) * line_height + 20
        panel_surface = pygame.Surface((300, panel_height))
        panel_surface.set_alpha(180)
        panel_surface.fill(BLACK)
        self.screen.blit(panel_surface, (hud_x - 5, hud_y - 5))

        for i, stat in enumerate(stats):
            color = status_color if i == len(stats) - 1 else WHITE
            text = self.font_medium.render(stat, True, color)
            self.screen.blit(text, (hud_x, hud_y + i * line_height))

        fps = self.clock.get_fps()
        runtime = time.time() - self.start_time
        perf_text = self.font_small.render(f"FPS: {fps:.1f} | Runtime: {runtime:.1f}s", True, GRAY)
        self.screen.blit(perf_text, (hud_x, hud_y + panel_height + 10))

    def draw_minimap(self, game_state, map_bounds, pickups=None):
        minimap_size = 150
        minimap_x = self.width - minimap_size - 10
        minimap_y = 10

        minimap_surface = pygame.Surface((minimap_size, minimap_size))
        minimap_surface.set_alpha(150)
        minimap_surface.fill((0, 20, 0))
        self.screen.blit(minimap_surface, (minimap_x, minimap_y))
        pygame.draw.rect(self.screen, GREEN, (minimap_x, minimap_y, minimap_size, minimap_size), 2)

        world_width = map_bounds['max_x'] - map_bounds['min_x']
        world_height = map_bounds['max_y'] - map_bounds['min_y']
        scale_x = minimap_size / world_width
        scale_y = minimap_size / world_height

        if pickups:
            for pickup in pickups:
                if not pickup['collected']:
                    px = minimap_x + (pickup['x'] - map_bounds['min_x']) * scale_x
                    py = minimap_y + (pickup['y'] - map_bounds['min_y']) * scale_y
                    pygame.draw.circle(self.screen, YELLOW, (int(px), int(py)), 2)

        ship_mx = minimap_x + (game_state['ship_x'] - map_bounds['min_x']) * scale_x
        ship_my = minimap_y + (game_state['ship_y'] - map_bounds['min_y']) * scale_y
        pygame.draw.circle(self.screen, CYAN, (int(ship_mx), int(ship_my)), 3)

    def draw_debug_path(self, debug_path):
        """Draw pathfinder debug path as a yellow dotted line."""
        if not debug_path or len(debug_path) < 2:
            return
        step = max(1, len(debug_path) // 100)
        points = []
        for i in range(0, len(debug_path), step):
            x, y = debug_path[i]
            sx, sy = self.world_to_screen(x, y)
            points.append((sx, sy))
        if len(debug_path) - 1 not in range(0, len(debug_path), step):
            x, y = debug_path[-1]
            sx, sy = self.world_to_screen(x, y)
            points.append((sx, sy))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, YELLOW, False, points, 2)

    def draw_mcts_debug(self, game_state, mcts_debug):
        """Draw MCTS debug overlay: heuristic breakdown, action stats, direction arrow, target crosshair."""
        if not mcts_debug:
            return

        h = mcts_debug.get("heuristic", {})
        t = mcts_debug.get("target", {})
        ship_sx, ship_sy = self.world_to_screen(game_state['ship_x'], game_state['ship_y'])

        # --- Target pickup crosshair (red X with label) ---
        if t.get("idx", -1) >= 0:
            tx, ty = self.world_to_screen(t["x"], t["y"])
            size = 15
            pygame.draw.line(self.screen, RED, (tx - size, ty - size), (tx + size, ty + size), 2)
            pygame.draw.line(self.screen, RED, (tx - size, ty + size), (tx + size, ty - size), 2)
            # Collection radius ring (46.5px)
            collection_r = int(46.5 * self.scale)
            pygame.draw.circle(self.screen, RED, (tx, ty), collection_r, 1)
            # Label
            label = self.font_small.render(
                f"Target #{t['idx']} path={t['path_dist']:.0f} euclid={t['euclidean_dist']:.0f}",
                True, RED)
            self.screen.blit(label, (tx + size + 4, ty - 8))

        # --- Direction arrow to target (cyan arrow from ship) ---
        dir_x = t.get("dir_x", h.get("dir_x", 0))
        dir_y = t.get("dir_y", h.get("dir_y", 0))
        if abs(dir_x) > 0.01 or abs(dir_y) > 0.01:
            arrow_len = 80
            end_x = ship_sx + dir_x * arrow_len
            end_y = ship_sy + dir_y * arrow_len
            pygame.draw.line(self.screen, CYAN, (ship_sx, ship_sy), (int(end_x), int(end_y)), 3)
            # Arrowhead
            perp_x, perp_y = -dir_y * 8, dir_x * 8
            tip_x, tip_y = int(end_x), int(end_y)
            base_x, base_y = end_x - dir_x * 12, end_y - dir_y * 12
            pygame.draw.polygon(self.screen, CYAN, [
                (tip_x, tip_y),
                (int(base_x + perp_x), int(base_y + perp_y)),
                (int(base_x - perp_x), int(base_y - perp_y)),
            ])

        # --- Heuristic breakdown panel (bottom-left) ---
        panel_x = 10
        panel_y = self.height - 240
        line_h = 18

        path_dist = h.get("path_dist", 0)
        speed_toward = h.get("speed_toward", 0)
        alignment = h.get("alignment", 0)
        min_tti = h.get("min_tti", float('inf'))
        tti_str = f"{min_tti:.2f}s" if min_tti < 100 else "safe"

        lines = [
            ("HEURISTIC BREAKDOWN", WHITE),
            (f"  Total:       {h.get('total', 0):+.1f}", WHITE),
            (f"  Pickups:     {h.get('pickups_score', 0):+.1f}", LIGHT_GREEN),
            (f"  Proximity:   {h.get('proximity_score', 0):+.1f}  (dist={path_dist:.0f})", YELLOW),
            (f"  Velocity:    {h.get('velocity_score', 0):+.1f}  (spd={speed_toward:.1f})", MAGENTA),
            (f"  Orientation: {h.get('orientation_score', 0):+.1f}  (align={alignment:.2f})", CYAN),
            (f"  TTI penalty: {h.get('tti_penalty', 0):+.1f}  (tti={tti_str})", RED if h.get('tti_penalty', 0) < 0 else GRAY),
            (f"  Target: #{t.get('idx', '?')} euclid={t.get('euclidean_dist', 0):.0f} path={t.get('path_dist', 0):.0f}", RED),
        ]

        panel_surface = pygame.Surface((340, len(lines) * line_h + 10))
        panel_surface.set_alpha(180)
        panel_surface.fill(BLACK)
        self.screen.blit(panel_surface, (panel_x - 5, panel_y - 5))

        for i, (text, color) in enumerate(lines):
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (panel_x, panel_y + i * line_h))

        # --- Action stats panel (bottom-right) ---
        action_stats = mcts_debug.get("action_stats", [])
        if action_stats:
            a_panel_x = self.width - 310
            a_panel_y = self.height - 170
            total_visits = sum(s["visits"] for s in action_stats)

            a_lines = [("MCTS ACTION VOTES", WHITE)]
            for s in action_stats:
                pct = (s["visits"] / total_visits * 100) if total_visits > 0 else 0
                bar_len = int(pct / 100 * 15)
                bar = "|" * bar_len
                a_lines.append((
                    f"  {s['name']:10s} {s['visits']:5d} ({pct:4.1f}%) {s['mean_value']:+8.1f}  {bar}",
                    GREEN if s == action_stats[0] else GRAY,
                ))

            a_surface = pygame.Surface((300, len(a_lines) * line_h + 10))
            a_surface.set_alpha(180)
            a_surface.fill(BLACK)
            self.screen.blit(a_surface, (a_panel_x - 5, a_panel_y - 5))

            for i, (text, color) in enumerate(a_lines):
                rendered = self.font_small.render(text, True, color)
                self.screen.blit(rendered, (a_panel_x, a_panel_y + i * line_h))

    def render_frame(self, game_state, info, reward, action=None, pickups=None, map_bounds=None, debug_path=None, mcts_debug=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                if not self.recording:
                    self.recording = True
                    self.record_log = []
                    print("[RECORD] Started recording MCTS decisions (press X again to stop & save)")
                else:
                    self.recording = False
                    self._save_recording()

        # Record debug data if active
        if self.recording and mcts_debug:
            action_names = ["COAST", "THRUST", "LEFT", "LEFT+THR", "RIGHT", "RIGHT+THR"]
            action_str = "NONE"
            if action is not None and len(action) == 3:
                idx = int(action[0]) * 4 + int(action[1]) * 2 + int(action[2])
                action_str = ["COAST", "THRUST", "LEFT", "LEFT+THR", "RIGHT", "RIGHT+THR", "L+R", "ALL"][idx]
            self.record_log.append({
                "frame": self.frame_count,
                "step": info.get("step_count", 0),
                "ship_x": round(game_state["ship_x"], 1),
                "ship_y": round(game_state["ship_y"], 1),
                "ship_vx": round(game_state["ship_vx"], 1),
                "ship_vy": round(game_state["ship_vy"], 1),
                "rotation_deg": round(math.degrees(game_state["ship_rotation"]) % 360, 1),
                "action": action_str,
                "heuristic": {k: round(v, 2) if isinstance(v, float) else v
                              for k, v in mcts_debug.get("heuristic", {}).items()},
                "target": mcts_debug.get("target", {}),
                "action_votes": mcts_debug.get("action_stats", []),
            })

        self.screen.fill(BLACK)
        self.update_camera(game_state['ship_x'], game_state['ship_y'])
        self.draw_grid()

        if map_bounds:
            self.draw_boundaries(map_bounds)
            self.draw_map_lines(map_bounds)
        if pickups:
            self.draw_pickups(pickups)
        if debug_path:
            self.draw_debug_path(debug_path)

        self.draw_ship(
            game_state['ship_x'], game_state['ship_y'], game_state['ship_rotation'],
            game_state['ship_vx'], game_state['ship_vy'],
            action is not None and len(action) > 2 and action[2] > 0,
            game_state.get('ship_exploded', False),
        )
        self.draw_hud(game_state, info, reward, action)
        self.draw_mcts_debug(game_state, mcts_debug)

        if map_bounds:
            self.draw_minimap(game_state, map_bounds, pickups)

        if self.recording:
            rec_text = self.font_medium.render("REC [X to stop]", True, RED)
            self.screen.blit(rec_text, (self.width // 2 - 60, 10))

        pygame.display.flip()
        self.frame_count += 1
        return True

    def _save_recording(self):
        import json
        path = "mcts_debug_recording.json"
        with open(path, "w") as f:
            json.dump(self.record_log, f, indent=2, default=str)
        print(f"[RECORD] Saved {len(self.record_log)} frames to {path}")

    def close(self):
        if self.recording and self.record_log:
            self._save_recording()
        pygame.quit()

    def wait_for_fps(self, target_fps=60):
        self.clock.tick(target_fps)
