import json

import numpy as np

from spaceace.ghost_actions import action_to_index, dump_action_file, load_action_file


def test_action_to_index_accepts_canonical_triplets():
    assert action_to_index(np.array([0, 1, 1], dtype=np.int32)) == 5
    assert action_to_index([1, 0, 0]) == 2


def test_action_file_round_trip_indices_and_raw_actions(tmp_path):
    path = tmp_path / "trace.json"
    dump_action_file(path, level=7, action_indices=[0, 3, 5], ticks=3)

    level, actions = load_action_file(path)
    assert level == 7
    assert actions == [0, 3, 5]

    payload = json.loads(path.read_text())
    assert payload["seconds"] == 0.05
    assert payload["raw_actions"] == [[0, 0, 0], [1, 0, 1], [0, 1, 1]]


def test_action_file_loads_raw_triplets(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"level": 6, "raw_actions": [[0, 0, 1], [0, 1, 0]]}))

    level, actions = load_action_file(path)
    assert level == 6
    assert actions == [1, 4]
