#!/usr/bin/env python3
"""
Play SpaceAce manually with keyboard controls.

Controls:
    Left Arrow  — rotate left
    Right Arrow — rotate right
    Up Arrow    — thrust
    R           — restart level
    Q / Escape  — quit

Usage:
    python play.py
    python play.py --level 1
"""

import argparse
import time

import numpy as np
import pygame

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.core.viz import VisualRenderer, extract_game_info


def play(level: int, max_steps: int):
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    renderer = VisualRenderer(width=1200, height=800)

    print(f"Playing level {level}")
    print("Controls: Arrow keys (left/right=rotate, up=thrust), R=restart, Q/Esc=quit")

    try:
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0

        while True:
            keys = pygame.key.get_pressed()
            action = np.array([
                int(keys[pygame.K_LEFT]),
                int(keys[pygame.K_RIGHT]),
                int(keys[pygame.K_UP]),
            ], dtype=np.int32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            game_state, pickups, map_bounds = extract_game_info(obs, info, env)
            game_state["level"] = level
            running = renderer.render_frame(game_state, info, total_reward, action, pickups, map_bounds)
            if not running:
                break
            renderer.wait_for_fps(60)

            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                break

            if terminated or truncated:
                status = "LEVEL COMPLETE!" if info.get("level_completed") else (
                    "CRASHED!" if info.get("ship_exploded") else "TIME UP!")
                print(f"{status}  steps={step_count}  reward={total_reward:.1f}")
                time.sleep(2)

                if keys[pygame.K_r] or terminated or truncated:
                    obs, info = env.reset()
                    total_reward = 0.0
                    step_count = 0

    finally:
        renderer.close()
        env.close()


def main():
    p = argparse.ArgumentParser(description="Play SpaceAce manually")
    p.add_argument("--level", type=int, default=0, help="Game level (default: 0)")
    p.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode (default: 5000)")
    args = p.parse_args()
    play(args.level, args.max_steps)


if __name__ == "__main__":
    main()
