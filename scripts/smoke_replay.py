"""Smoke check: the solved L7 tape replays tick-exactly in the solver sim."""
import json

import spaceace_rl

d = json.load(open("ghost_actions/L7_tas.json"))
s = spaceace_rl.PySolver(7)
completed, crashed, ticks = s.replay(bytes(d["actions"]))
assert completed and not crashed, f"tape failed: completed={completed} crashed={crashed}"
assert ticks == d["ticks"], f"tick mismatch: solver={ticks} sidecar={d['ticks']}"

# Insert one known-redundant late action, then require both improvement APIs
# to recover the canonical one-tick-shorter tape even at width zero. This
# exercises reference injection, parent-link reconstruction, and exact suffix
# acceptance without adding a meaningful smoke-test cost.
base = list(d["actions"])
pos = len(base) - 3
padded = base[:pos] + [0] + base[pos:]
pad_completed, pad_crashed, pad_ticks = s.replay(bytes(padded))
assert pad_completed and not pad_crashed and pad_ticks == ticks + 1

window = s.resolve_window_exact(
    bytes(padded), pos - 10, pos + 1,
    save_ticks=1, width=0, seed=7,
)
assert window is not None
w_completed, w_crashed, w_ticks = s.replay(bytes(window))
assert w_completed and not w_crashed and w_ticks == len(window) == ticks

suffix = s.resolve_suffix(bytes(padded), pos - 10, width=0, seed=11)
assert suffix is not None
s_completed, s_crashed, s_ticks = s.replay(bytes(suffix))
assert s_completed and not s_crashed and s_ticks == len(suffix) == ticks
print(f"L7 tape: {ticks} ticks ({ticks / 60:.2f}s)")
