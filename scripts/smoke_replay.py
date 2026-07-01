"""Smoke check: the solved L7 tape replays tick-exactly in the solver sim."""
import json

import spaceace_rl

d = json.load(open("ghost_actions/L7_tas.json"))
s = spaceace_rl.PySolver(7)
completed, crashed, ticks = s.replay(bytes(d["actions"]))
assert completed and not crashed, f"tape failed: completed={completed} crashed={crashed}"
assert ticks == d["ticks"], f"tick mismatch: solver={ticks} sidecar={d['ticks']}"
print(f"L7 tape: {ticks} ticks ({ticks / 60:.2f}s)")
