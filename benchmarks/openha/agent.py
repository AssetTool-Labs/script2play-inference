"""script2play's client as an OpenHA agent for run.py: one model call per 200 ms, played as four
MineStudio ticks.

It behaves as the agent the reported OpenHA scores were measured with: the 640x360 POV sent at
its own size as a JPEG at quality 88, the brief "minecraft" system prompt with MineStudio's task
description as it is, an answer that is not exactly the 4-step grammar played as idle (and
counted), the mouse spread evenly over the four ticks at 0.15 deg/px in the world and at the
cursor gain MineStudio recorded with a screen open. Only history 0 and the native frame size are
supported, as the 4B was run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script2play import VLAClient, actions, encode_jpeg, parse  # noqa: E402

TICKS = 4
DEG_PER_PX = (0.15, 0.15)
# x then y, with a screen open: the cursor gain measured on MineStudio's mc_scripted_gui recordings.
GUI_DEG_PER_PX = (0.144, 0.145)

KEY_TO_ENV = {"W": "forward", "S": "back", "A": "left", "D": "right", "Space": "jump",
              "Shift": "sneak", "Ctrl": "sprint", "LB": "attack", "RB": "use",
              "Q": "drop", "E": "inventory"}
KEY_TO_ENV.update({name: f"hotbar.{i}" for i, name in enumerate(
    "one two three four five six seven eight nine".split(), 1)})
ENV_BUTTONS = ("attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use",
               "drop", "inventory") + tuple(f"hotbar.{i}" for i in range(1, 10))


def noop() -> dict:
    return {**{b: 0 for b in ENV_BUTTONS}, "camera": [0.0, 0.0]}


def env_actions(action: dict | None, gui: bool) -> list[dict]:
    """A decoded chunk as TICKS MineStudio env actions. None (unparseable) plays as idle."""
    if action is None:
        return [noop() for _ in range(TICKS)]
    gx, gy = GUI_DEG_PER_PX if gui else DEG_PER_PX
    yaw, pitch = action["dx"] * gx / TICKS, action["dy"] * gy / TICKS
    out = []
    for k in range(TICKS):
        a = noop()
        for name in action["chunks"][k] if k < len(action["chunks"]) else ():
            if name in KEY_TO_ENV:
                a[KEY_TO_ENV[name]] = 1
        a["camera"] = [pitch, yaw]
        out.append(a)
    return out


class QwenKeyStringAgent:
    """run.py's agent: ``reset(instruction)`` per episode, ``act(pov, gui)`` per decision."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "script2play",
                 max_tokens: int = 48, temperature: float = 0.8, top_p: float = 0.99,
                 timeout: float = 60.0, block_keys: tuple = (), history: int = 0,
                 image_size: tuple | None = None, checkpoint: str | None = None):
        if history or image_size:
            raise ValueError("this agent plays history 0 at the native frame size only")
        import os
        checkpoint = checkpoint or os.environ["SCRIPT2PLAY_CHECKPOINT"]
        self.client = VLAClient(base_url, checkpoint, game="minecraft", temperature=temperature, top_p=top_p,
                                max_tokens=max_tokens, timeout=timeout, resize_width=None, normalize_task=False)
        self.block_keys = set(block_keys)
        self.instruction = None
        self.calls = self.failures = 0
        self.latency = 0.0
        self.last_jpeg = None

    def reset(self, instruction: str | None = None) -> None:
        self.instruction = instruction

    def generate(self, pov) -> str:
        self.last_jpeg = encode_jpeg(pov, None)
        body = self.client.request(self.last_jpeg, self.instruction or "")
        if not self.instruction:
            body["messages"] = body["messages"][1:]
        t0 = time.perf_counter()
        resp = self.client.session.post(f"{self.client.base_url}/chat/completions", json=body,
                                        timeout=self.client.timeout)
        resp.raise_for_status()
        self.latency += time.perf_counter() - t0
        self.calls += 1
        return resp.json()["choices"][0]["message"]["content"] or ""

    def act(self, pov, gui: bool) -> tuple[str, list[dict]]:
        """(the generated text, the TICKS env actions it plays as). Keys in block_keys are
        generated but not played."""
        text = self.generate(pov)
        try:
            actions.validate(text, TICKS)
            chunk = parse(text)
            action = {"dx": chunk.dx, "dy": chunk.dy, "dz": chunk.dz, "chunks": chunk.keys}
        except ValueError:
            self.failures += 1
            action = None
        if action is not None and self.block_keys:
            action = {**action, "chunks": [[k for k in c if k not in self.block_keys] for c in action["chunks"]]}
        return text, env_actions(action, gui)
