#!/usr/bin/env python3
"""Render every generated level as a top-down overview PNG.

Usage:
    uv run python scripts/screenshot_levels.py                # all generated levels (3000+)
    uv run python scripts/screenshot_levels.py --range 4000-4099
    uv run python scripts/screenshot_levels.py --range 3000-3049 --out gallery/curriculum
"""

import argparse
import json
import math
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"  # headless pygame

import pygame
import spaceace_rl

# Colors (matching viz.py palette)
BLACK = (0, 0, 0)
GREEN = (0, 255, 65)
DARK_GREEN = (0, 204, 51)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GRAY = (60, 60, 60)


def render_level(level: int, width: int = 1200, height: int = 900) -> pygame.Surface:
    """Render a full top-down overview of a level to a pygame Surface."""
    game = spaceace_rl.PyGameInstance(level, 3000)
    game.reset()
    geom = dict(game.get_map_geometry())

    map_lines = [(l[0], l[1], l[2], l[3]) for l in geom.get("map_lines", []) if len(l) >= 4]
    pickups = [(p[0], p[1]) for p in geom.get("pickup_positions", []) if len(p) >= 3]
    b = geom.get("bounds", {})
    min_x, max_x = b.get("min_x", 0), b.get("max_x", 1200)
    min_y, max_y = b.get("min_y", 0), b.get("max_y", 800)

    # Get ship spawn from observation
    obs = list(game.get_observation())
    ship_x, ship_y = obs[0], obs[1]

    # Compute scale to fit the level with padding
    world_w = max_x - min_x
    world_h = max_y - min_y
    if world_w == 0 or world_h == 0:
        world_w = max(world_w, 1)
        world_h = max(world_h, 1)

    header_h = 40  # space for title
    draw_h = height - header_h
    pad = 40
    scale_x = (width - 2 * pad) / world_w
    scale_y = (draw_h - 2 * pad) / world_h
    scale = min(scale_x, scale_y)

    # Center the level in the available area
    rendered_w = world_w * scale
    rendered_h = world_h * scale
    offset_x = pad + (width - 2 * pad - rendered_w) / 2
    offset_y = header_h + pad + (draw_h - 2 * pad - rendered_h) / 2

    def world_to_screen(wx, wy):
        sx = int((wx - min_x) * scale + offset_x)
        sy = int((wy - min_y) * scale + offset_y)
        return sx, sy

    surface = pygame.Surface((width, height))
    surface.fill(BLACK)

    # Grid
    grid_world = 50
    grid_px = grid_world * scale
    if grid_px >= 8:
        gx = min_x - (min_x % grid_world)
        while gx <= max_x:
            sx, _ = world_to_screen(gx, min_y)
            _, sy0 = world_to_screen(0, min_y)
            _, sy1 = world_to_screen(0, max_y)
            pygame.draw.line(surface, (0, 25, 0), (sx, sy0), (sx, sy1), 1)
            gx += grid_world
        gy = min_y - (min_y % grid_world)
        while gy <= max_y:
            _, sy = world_to_screen(min_x, gy)
            sx0, _ = world_to_screen(min_x, 0)
            sx1, _ = world_to_screen(max_x, 0)
            pygame.draw.line(surface, (0, 25, 0), (sx0, sy), (sx1, sy), 1)
            gy += grid_world

    # Walls
    for x1, y1, x2, y2 in map_lines:
        s1 = world_to_screen(x1, y1)
        s2 = world_to_screen(x2, y2)
        pygame.draw.line(surface, GREEN, s1, s2, max(2, int(2 * scale / 0.8)))
        pygame.draw.line(surface, DARK_GREEN, s1, s2, 1)

    # Pickups
    for i, (px, py) in enumerate(pickups):
        sx, sy = world_to_screen(px, py)
        r = max(6, int(12 * scale / 0.8))
        pygame.draw.circle(surface, CYAN, (sx, sy), r)
        pygame.draw.circle(surface, WHITE, (sx, sy), r, 1)
        # Label
        font_sm = pygame.font.Font(None, max(14, int(16 * scale / 0.5)))
        label = font_sm.render(str(i), True, WHITE)
        surface.blit(label, (sx + r + 2, sy - label.get_height() // 2))

    # Ship spawn
    sx, sy = world_to_screen(ship_x, ship_y)
    size = max(8, int(12 * scale / 0.8))
    pygame.draw.polygon(surface, YELLOW, [
        (sx, sy - size),
        (sx - size * 0.6, sy + size * 0.6),
        (sx + size * 0.6, sy + size * 0.6),
    ])
    font_sm = pygame.font.Font(None, max(14, int(16 * scale / 0.5)))
    spawn_label = font_sm.render("SPAWN", True, YELLOW)
    surface.blit(spawn_label, (sx + size + 2, sy - spawn_label.get_height() // 2))

    # Title bar
    font_title = pygame.font.Font(None, 32)
    title = font_title.render(
        f"Level {level}   |   {int(world_w)}x{int(world_h)}px   |   "
        f"{len(pickups)} pickup{'s' if len(pickups) != 1 else ''}   |   "
        f"{len(map_lines)} walls",
        True, WHITE,
    )
    surface.blit(title, (10, 8))

    return surface


def parse_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def main():
    parser = argparse.ArgumentParser(description="Screenshot every generated level")
    parser.add_argument("--range", type=str, default=None,
                        help="Level range, e.g. 4000-4099 or 3000-3049 (default: all >=3000)")
    parser.add_argument("--out", type=str, default="screenshots/levels",
                        help="Output directory (default: screenshots/levels)")
    parser.add_argument("--width", type=int, default=1200, help="Image width (default: 1200)")
    parser.add_argument("--height", type=int, default=900, help="Image height (default: 900)")
    args = parser.parse_args()

    pygame.init()
    # Need a display for font rendering even in dummy mode
    pygame.display.set_mode((1, 1))

    # Determine which levels to render
    if args.range:
        levels = parse_range(args.range)
    else:
        levels_path = os.path.join(os.path.dirname(__file__), "..", "data", "spaceace_levels.json")
        levels_path = os.path.normpath(levels_path)
        with open(levels_path) as f:
            all_levels = json.load(f)
        levels = sorted(int(k) for k in all_levels if not k.startswith("_") and int(k) >= 3000)

    os.makedirs(args.out, exist_ok=True)

    print(f"Rendering {len(levels)} levels to {args.out}/")
    for i, level in enumerate(levels):
        try:
            surface = render_level(level, args.width, args.height)
            path = os.path.join(args.out, f"level_{level:05d}.png")
            pygame.image.save(surface, path)
            if (i + 1) % 25 == 0 or i == 0 or i == len(levels) - 1:
                print(f"  [{i + 1}/{len(levels)}] {path}")
        except Exception as e:
            print(f"  [{i + 1}/{len(levels)}] Level {level}: FAILED ({e})")

    pygame.quit()
    print(f"Done. {len(levels)} screenshots saved to {args.out}/")


if __name__ == "__main__":
    main()
