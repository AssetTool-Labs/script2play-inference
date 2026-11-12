import importlib.util
from pathlib import Path

import numpy as np
import pytest

from script2play import parse

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vizdoom_chunk_to_tics():
    ev = load("benchmarks/vizdoom/eval_gamewam.py")
    policy = object.__new__(ev.DoomPolicy)
    policy.stats = {"dropped_keys": {}}
    plan, debug = policy.translate(parse("70 5 0 ; LB W ; LB ; Right ; E"))
    assert len(plan) == 7 and all(v.shape == (9,) for v in plan)
    # Groups on tics 0-1, 2-3, 4-5, 6: shoot and forward, shoot, the Right arrow, a key ViZDoom lacks.
    assert [int(v[0]) for v in plan] == [1, 1, 1, 1, 0, 0, 0]
    assert [int(v[1]) for v in plan] == [1, 1, 0, 0, 0, 0, 0]
    mouse = 70 * 0.15 / 7 / 10
    arrow = 120 / 35 / 10
    np.testing.assert_allclose([v[8] for v in plan], [mouse] * 4 + [mouse + arrow] * 2 + [mouse], rtol=1e-6)
    assert debug["dropped_keys"] == ["E"]


def test_openha_chunk_to_ticks():
    ag = load("benchmarks/openha/agent.py")
    c = parse("40 -8 0 ; W LB ; ; E ; one")
    action = {"dx": c.dx, "dy": c.dy, "dz": c.dz, "chunks": c.keys}
    world, gui = ag.env_actions(action, False), ag.env_actions(action, True)
    assert [a["forward"] for a in world] == [1, 0, 0, 0] and world[2]["inventory"] == 1 and world[3]["hotbar.1"] == 1
    assert world[0]["camera"] == pytest.approx([-8 * 0.15 / 4, 40 * 0.15 / 4])
    assert gui[0]["camera"] == pytest.approx([-8 * 0.145 / 4, 40 * 0.144 / 4])
    assert ag.env_actions(None, False) == [ag.noop()] * 4
