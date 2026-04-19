#!/usr/bin/env python3
"""
TAS (tool-assisted speedrun) interface for SpaceAce.

Record, pause, rewind, frame-step and redo sections to craft an optimal run.

Controls
  Gameplay (while LIVE):
    Left/Right  — rotate
    Up          — thrust
  Timeline:
    Space       — pause / resume
    Backspace   — rewind (hold) while paused or live
    , / .       — step back / step forward 1 frame (paused)
    < / >       — step back / step forward 10 frames (paused)
    Home        — jump to start
    End         — jump to end of recorded timeline
  Session:
    R           — reset level (clears timeline)
    S           — save TAS to JSON
    L           — load TAS from JSON and replay to end
    Tab         — cycle speed: 60 / 30 / 15 fps
    Q / Esc     — quit

When paused, pressing . replays the recorded action for that frame.
When you un-pause and provide input, recorded future is truncated and
a new branch is recorded from the current cursor forward.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pygame

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.core.viz import VisualRenderer, extract_game_info


SPEEDS = [60, 30, 15]


class Timeline:
    """Ring of (snapshot, action) entries. snapshots[i] is the state BEFORE action[i]."""

    def __init__(self, env: SpaceAceDirectEnv):
        self.env = env
        self.snapshots = [env.save_state()]
        self.actions: list[list[int]] = []
        self.cursor = 0  # number of steps from start; matches index into actions

    def __len__(self) -> int:
        return len(self.actions)

    def reset(self):
        self.env.reset()
        self.snapshots = [self.env.save_state()]
        self.actions = []
        self.cursor = 0

    def step_live(self, action: list[int]):
        """Advance one frame. Truncates any future beyond cursor before recording."""
        if self.cursor < len(self.actions):
            # Branching: discard future
            self.actions = self.actions[: self.cursor]
            self.snapshots = self.snapshots[: self.cursor + 1]

        self.env.load_state(self.snapshots[self.cursor])
        obs, reward, terminated, truncated, info = self.env.step(np.array(action, dtype=np.int32))
        self.actions.append(list(action))
        self.snapshots.append(self.env.save_state())
        self.cursor += 1
        return obs, reward, terminated, truncated, info

    def step_forward_recorded(self):
        """Replay one recorded action forward; no-op at end."""
        if self.cursor >= len(self.actions):
            return None
        self.env.load_state(self.snapshots[self.cursor])
        action = self.actions[self.cursor]
        obs, reward, terminated, truncated, info = self.env.step(np.array(action, dtype=np.int32))
        self.cursor += 1
        # snapshot already stored; no need to overwrite (deterministic)
        return obs, reward, terminated, truncated, info, action

    def seek(self, frame: int):
        frame = max(0, min(frame, len(self.actions)))
        self.cursor = frame
        self.env.load_state(self.snapshots[frame])

    def step_back(self, n: int = 1):
        self.seek(self.cursor - n)

    def step_forward(self, n: int = 1):
        for _ in range(n):
            if self.cursor >= len(self.actions):
                break
            self.step_forward_recorded()

    def save(self, path: Path, level: int):
        path.write_text(json.dumps({"level": level, "actions": self.actions}))

    def load(self, path: Path):
        data = json.loads(path.read_text())
        self.reset()
        for a in data["actions"]:
            self.step_live(list(a))
        return data.get("level")


def draw_timeline_bar(renderer: VisualRenderer, timeline: Timeline, paused: bool, speed_fps: int):
    surf = renderer.screen
    w, h = renderer.width, renderer.height
    bar_h = 28
    margin = 10
    bar_rect = pygame.Rect(margin, h - bar_h - margin, w - 2 * margin, bar_h)
    pygame.draw.rect(surf, (20, 20, 20), bar_rect)
    pygame.draw.rect(surf, (0, 200, 50), bar_rect, 1)

    total = max(len(timeline), 1)
    frac = timeline.cursor / total
    fill_w = int(bar_rect.width * frac)
    pygame.draw.rect(surf, (0, 150, 40), (bar_rect.x, bar_rect.y, fill_w, bar_rect.height))

    mode = "PAUSED" if paused else "LIVE"
    color = (255, 255, 0) if paused else (0, 255, 65)
    label = renderer.font_small.render(
        f"{mode}  frame {timeline.cursor}/{len(timeline)}   {speed_fps} fps",
        True, color,
    )
    surf.blit(label, (bar_rect.x + 8, bar_rect.y + 6))


def action_from_keys(keys) -> list[int]:
    return [
        int(keys[pygame.K_LEFT]),
        int(keys[pygame.K_RIGHT]),
        int(keys[pygame.K_UP]),
    ]


def run(level: int, max_steps: int, save_path: Path):
    env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
    renderer = VisualRenderer(width=1200, height=800)
    env.reset()
    timeline = Timeline(env)

    paused = False
    speed_idx = 0
    last_info: dict = env.get_info()
    last_action = [0, 0, 0]
    total_reward = 0.0

    print(__doc__)

    running = True
    try:
        while running:
            # Handle discrete events first (consumes queue before renderer).
            step_back_once = False
            step_fwd_once = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    mods = pygame.key.get_mods()
                    shift = bool(mods & pygame.KMOD_SHIFT)
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif k == pygame.K_SPACE:
                        paused = not paused
                    elif k == pygame.K_TAB:
                        speed_idx = (speed_idx + 1) % len(SPEEDS)
                    elif k == pygame.K_r:
                        timeline.reset()
                        total_reward = 0.0
                        last_info = env.get_info()
                    elif k == pygame.K_s:
                        timeline.save(save_path, level)
                        print(f"[TAS] saved {len(timeline)} frames -> {save_path}")
                    elif k == pygame.K_l:
                        if save_path.exists():
                            loaded_level = timeline.load(save_path)
                            print(f"[TAS] loaded {len(timeline)} frames from {save_path} (level {loaded_level})")
                            last_info = env.get_info()
                        else:
                            print(f"[TAS] no file at {save_path}")
                    elif k == pygame.K_HOME:
                        timeline.seek(0)
                    elif k == pygame.K_END:
                        timeline.seek(len(timeline))
                    elif k == pygame.K_COMMA:
                        timeline.step_back(10 if shift else 1)
                    elif k == pygame.K_PERIOD:
                        step_fwd_once = True
                        if shift:
                            timeline.step_forward(10)
                            step_fwd_once = False

            keys = pygame.key.get_pressed()

            # Rewind: holding backspace scrubs backward at speed.
            if keys[pygame.K_BACKSPACE]:
                timeline.step_back(2)

            # Advance the sim.
            if not paused:
                action = action_from_keys(keys)
                # If user is providing input at a live cursor that matches recorded
                # future, let recorded future play unless they press something.
                any_input = any(action)
                at_end = timeline.cursor >= len(timeline)
                if at_end or any_input:
                    obs, reward, terminated, truncated, info = timeline.step_live(action)
                    total_reward += reward
                    last_info = info
                    last_action = action
                    if terminated or truncated:
                        paused = True
                else:
                    result = timeline.step_forward_recorded()
                    if result is not None:
                        obs, reward, terminated, truncated, info, action = result
                        total_reward += reward
                        last_info = info
                        last_action = action
            else:
                if step_fwd_once:
                    result = timeline.step_forward_recorded()
                    if result is not None:
                        _, _, _, _, info, action = result
                        last_info = info
                        last_action = action

            # Render from current env state.
            obs = env.get_observation()
            info = env.get_info()
            game_state, pickups, map_bounds = extract_game_info(obs, info, env)
            game_state["level"] = level
            action_arr = np.array(last_action, dtype=np.int32)
            alive = renderer.render_frame(game_state, info, total_reward, action_arr, pickups, map_bounds)
            if not alive:
                running = False

            draw_timeline_bar(renderer, timeline, paused, SPEEDS[speed_idx])
            pygame.display.flip()

            renderer.wait_for_fps(SPEEDS[speed_idx])
    finally:
        renderer.close()
        env.close()


def main():
    p = argparse.ArgumentParser(description="TAS interface for SpaceAce")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save", type=str, default="tas_run.json", help="Path used by S/L hotkeys")
    args = p.parse_args()
    run(args.level, args.max_steps, Path(args.save))


if __name__ == "__main__":
    main()
