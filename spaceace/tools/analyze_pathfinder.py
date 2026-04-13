#!/usr/bin/env python3
"""
Visualize the pathfinder's distance field and routing decisions.
Renders the BFS/Dijkstra grid, blocked cells, pickup targets, and direction arrows
so you can see exactly what the pathfinder thinks.

Usage:
    uv run python analyze_pathfinder.py --level 6
    uv run python analyze_pathfinder.py --level 6 --ship-x -500 --ship-y -90 --collected 0,1,3,4,5
"""

import argparse
import math

import numpy as np
import pygame

import spaceace_rl


def parse_args():
    p = argparse.ArgumentParser(description="Visualize pathfinder state")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--ship-x", type=float, default=None, help="Override ship X position")
    p.add_argument("--ship-y", type=float, default=None, help="Override ship Y position")
    p.add_argument("--collected", type=str, default=None,
                   help="Comma-separated indices of collected pickups (e.g. '0,1,3,4,5')")
    return p.parse_args()


def main():
    args = parse_args()

    # Set up game
    g = spaceace_rl.PyGameInstance(args.level, 3000)
    g.reset()
    mcts = spaceace_rl.PyMCTSEngine(args.level, 3000)

    geom = g.get_map_geometry()
    pickups = geom["pickup_positions"]
    map_lines = geom["map_lines"]
    bounds = geom["bounds"]
    obs = g.get_observation()

    ship_x = args.ship_x if args.ship_x is not None else float(obs[0])
    ship_y = args.ship_y if args.ship_y is not None else float(obs[1])

    # Figure out collected state
    num_pickups = len(pickups)
    if args.collected:
        collected_indices = set(int(x) for x in args.collected.split(","))
    else:
        collected_indices = set()

    print(f"Level {args.level}: {num_pickups} pickups")
    print(f"Ship position: ({ship_x:.1f}, {ship_y:.1f})")
    print(f"Collected: {collected_indices}")
    print()

    # Get pathfinder info from current state
    state = g.save_state()
    path_dist, dir_x, dir_y = mcts.get_pathfinder_stats(state)
    debug_path = mcts.get_debug_path(state)

    print(f"From default state:")
    print(f"  pathDist={path_dist:.1f}, dir=({dir_x:.3f}, {dir_y:.3f})")
    print(f"  debug_path: {len(debug_path)} points")
    if debug_path:
        print(f"  path start: ({debug_path[0][0]:.1f}, {debug_path[0][1]:.1f})")
        print(f"  path end:   ({debug_path[-1][0]:.1f}, {debug_path[-1][1]:.1f})")

    # Show all pickups with distances
    print()
    print("Pickups:")
    for i, (px, py, collected) in enumerate(pickups):
        eucl = math.sqrt((px - ship_x)**2 + (py - ship_y)**2)
        status = "COLLECTED" if collected else ("SKIP" if i in collected_indices else "AVAILABLE")
        marker = " <-- path target?" if debug_path and math.sqrt(
            (px - debug_path[-1][0])**2 + (py - debug_path[-1][1])**2) < 30 else ""
        print(f"  [{i}] ({px:8.1f}, {py:8.1f}) eucl={eucl:6.1f} {status}{marker}")

    # --- Pygame visualization ---
    WIDTH, HEIGHT = 1400, 900
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pathfinder Analysis - Level {args.level}")
    font = pygame.font.Font(None, 20)
    font_large = pygame.font.Font(None, 28)

    # Compute world-to-screen transform (fit entire map)
    world_w = bounds["max_x"] - bounds["min_x"]
    world_h = bounds["max_y"] - bounds["min_y"]
    margin = 50
    scale = min((WIDTH - 2*margin) / world_w, (HEIGHT - 2*margin) / world_h)
    offset_x = margin - bounds["min_x"] * scale
    offset_y = margin - bounds["min_y"] * scale

    def w2s(wx, wy):
        return int(wx * scale + offset_x), int(wy * scale + offset_y)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        screen.fill((0, 0, 0))

        # Draw map lines
        for line in map_lines:
            x1, y1, x2, y2 = line
            pygame.draw.line(screen, (0, 100, 0), w2s(x1, y1), w2s(x2, y2), 1)

        # Draw map bounds
        corners = [w2s(bounds["min_x"], bounds["min_y"]),
                   w2s(bounds["max_x"], bounds["min_y"]),
                   w2s(bounds["max_x"], bounds["max_y"]),
                   w2s(bounds["min_x"], bounds["max_y"])]
        pygame.draw.lines(screen, (0, 80, 0), True, corners, 1)

        # Draw pathfinder debug path
        if debug_path and len(debug_path) >= 2:
            path_screen = [w2s(x, y) for x, y in debug_path]
            pygame.draw.lines(screen, (255, 255, 0), False, path_screen, 2)

        # Draw pickups
        for i, (px, py, collected) in enumerate(pickups):
            sx, sy = w2s(px, py)
            if collected or i in collected_indices:
                pygame.draw.circle(screen, (80, 80, 80), (sx, sy), 6, 1)
            else:
                pygame.draw.circle(screen, (0, 255, 255), (sx, sy), 8)
                pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 3)
            label = font.render(str(i), True, (200, 200, 200))
            screen.blit(label, (sx + 10, sy - 8))

        # Draw ship position
        sx, sy = w2s(ship_x, ship_y)
        pygame.draw.circle(screen, (0, 255, 0), (sx, sy), 6)

        # Draw pathfinder direction arrow from ship
        arrow_len = 40
        ax = sx + dir_x * arrow_len
        ay = sy + dir_y * arrow_len
        pygame.draw.line(screen, (255, 0, 255), (sx, sy), (int(ax), int(ay)), 3)

        # Draw direct lines to uncollected pickups (for comparison)
        for i, (px, py, collected) in enumerate(pickups):
            if not collected and i not in collected_indices:
                tx, ty = w2s(px, py)
                pygame.draw.line(screen, (50, 50, 50), (sx, sy), (tx, ty), 1)

        # HUD
        texts = [
            f"Level {args.level} | Ship: ({ship_x:.0f}, {ship_y:.0f})",
            f"PathDist: {path_dist:.1f} | Dir: ({dir_x:.3f}, {dir_y:.3f})",
            f"Collected: {sorted(collected_indices)} | Remaining: {num_pickups - len(collected_indices)}",
            f"Path end: ({debug_path[-1][0]:.0f}, {debug_path[-1][1]:.0f})" if debug_path else "No path",
            "",
            "Yellow = pathfinder route | Magenta = direction arrow",
            "Gray lines = direct to pickups | Cyan = uncollected pickups",
            "Q/Esc to quit",
        ]
        for i, t in enumerate(texts):
            surf = font.render(t, True, (200, 200, 200))
            screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()
        pygame.time.wait(50)

    pygame.quit()


if __name__ == "__main__":
    main()
