"""Smoke check: a small beam solve completes level 7 and replays."""
import spaceace_rl

s = spaceace_rl.PySolver(7)
tape = s.solve(width=8000, max_ticks=2400, seed=0, mix=1.0, proj_div=300.0)
assert tape is not None, "beam found no completion"
completed, crashed, ticks = s.replay(bytes(tape))
assert completed, "solution tape does not replay"
print(f"solved L7 in {ticks} ticks ({ticks / 60:.2f}s)")
