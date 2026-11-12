"""The action grammar the model writes, parsed robustly for serving.

The model emits one line per observation window — one *action chunk*::

    <|action_start|>dx dy dz ; K1 ; K2 ; … ; KN<|action_end|>

``dx dy`` are the mouse displacement summed over the window and ``dz`` the scroll
steps. Each ``Ki`` is one **action**: the keys held during one N-th of the window
(empty when nothing is held, and an action may name several keys at once). The N
actions together make the chunk, so N is the chunk size (4 for these checkpoints).
Keys repeated across consecutive actions are held; keys absent are released.

  * **``decode`` never raises**, so a control loop degrades to a null action:
    unknown key names are dropped and a malformed mouse field becomes 0 0 0.
    ``validate`` raises instead, for callers that treat anything but the exact
    grammar as idle (the OpenHA agent).
  * **Delimiters and a thinking block are tolerated.** ``skip_special_tokens``
    strips ``<|action_start|>`` / ``<|action_end|>`` upstream, but they (and a
    leading ``<think>…</think>``) may also be present, so the decoder is correct
    however the endpoint is configured.
"""

from __future__ import annotations

import re

# Key codes -> action-string names, as the training data wrote them.
KEYS = {
    8: "Back", 9: "Tab", 13: "Enter", 27: "Esc", 32: "Space",
    39: "Quote", 46: "Dot", 47: "Slash", 96: "Grave", 256: "Pad",
    273: "Up", 274: "Down", 275: "Right", 276: "Left",
    303: "Shift", 304: "Shift", 306: "Ctrl", 310: "Meta",
}
KEYS.update({48 + i: n for i, n in enumerate(
    "zero one two three four five six seven eight nine".split())})
KEYS.update({97 + i: chr(65 + i) for i in range(26)})

BUTTONS = {1: "LB", 2: "RB", 3: "MB"}

MAX_KEYS = 4        # simultaneous keys one action may name
LIMIT = 999         # mouse values clip here

KNOWN_NAMES = set(KEYS.values()) | set(BUTTONS.values())

_NUM = re.compile(r"^-?\d+$")
ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"


def _clip(v: int) -> int:
    return max(-LIMIT, min(LIMIT, int(v)))


def strip_wrappers(s: str) -> str:
    """Remove a leading thinking block and the action delimiters, if present.

    Correct whether the endpoint served the raw text (delimiters present) or set
    ``skip_special_tokens=True`` (delimiters already gone)."""
    if "</think>" in s:
        s = s.split("</think>", 1)[1]
    if ACTION_START in s:
        s = s.split(ACTION_START, 1)[1]
    if ACTION_END in s:
        s = s.split(ACTION_END, 1)[0]
    return s.strip()


def _mouse_values(head: str):
    nums = head.split()
    if len(nums) == 3 and all(_NUM.match(n) for n in nums):
        return tuple(_clip(int(n)) for n in nums)
    return None


def decode(s: str) -> dict:
    """Parse an action chunk into ``{"dx","dy","dz","actions":[[names],…]}``.

    Never raises: a malformed mouse field degrades to ``0 0 0``, unknown key
    names are dropped, and empty input yields zero actions. ``actions`` has one
    entry per action the model emitted (N = the chunk size), each a list of known
    key/button names truncated to ``MAX_KEYS``."""
    s = strip_wrappers(s)
    if not s:
        return {"dx": 0, "dy": 0, "dz": 0, "actions": []}

    fields = [f.strip() for f in s.split(";")]
    mouse = _mouse_values(fields[0])
    dx, dy, dz = mouse if mouse is not None else (0, 0, 0)

    actions = []
    for f in fields[1:]:
        names = [n for n in f.split() if n in KNOWN_NAMES][:MAX_KEYS]
        actions.append(names)
    return {"dx": dx, "dy": dy, "dz": dz, "actions": actions}


# Diagnostic flags reported by diagnose(). Ordered roughly worst-first.
FLAG_EMPTY_OUTPUT = "empty_output"
FLAG_NOT_ACTION_GRAMMAR = "not_action_grammar"
FLAG_BAD_MOUSE = "bad_mouse"
FLAG_NO_ACTIONS = "no_actions"
FLAG_UNKNOWN_KEYS = "unknown_keys"
FLAG_IDLE = "idle"

# Flags that mean "this text is not a usable action", as opposed to a usable
# action that happens to do nothing (FLAG_IDLE).
MALFORMED_FLAGS = frozenset(
    {FLAG_EMPTY_OUTPUT, FLAG_NOT_ACTION_GRAMMAR, FLAG_BAD_MOUSE, FLAG_NO_ACTIONS}
)


def diagnose(s: str) -> dict:
    """Explain what is wrong with a raw model output, without changing it.

    ``decode`` deliberately never raises — it degrades a malformed mouse field
    to ``0 0 0`` and silently drops unknown key names — which makes a model
    emitting prose indistinguishable from a model that chose to stand still.
    Both decode to an idle action. This reports the difference so the agent can
    warn instead of quietly holding still forever.

    Returns ``{"flags": [...], "ok": bool, "malformed": bool, "idle": bool,
    "unknown_keys": [...], "action_count": int}``. ``ok`` means the text parsed
    as the action grammar; an ``ok`` action may still be ``idle``."""
    flags: list[str] = []
    stripped = strip_wrappers(s or "")
    if not stripped:
        flags.append(FLAG_EMPTY_OUTPUT)
        return {
            "flags": flags,
            "ok": False,
            "malformed": True,
            "idle": True,
            "unknown_keys": [],
            "action_count": 0,
            "text": "",
        }

    fields = [f.strip() for f in stripped.split(";")]
    mouse = _mouse_values(fields[0])
    mouse_ok = mouse is not None
    if not mouse_ok:
        # No numeric triple and no separators at all is the captioning-fallback
        # signature: the model answered in prose instead of the action grammar.
        flags.append(
            FLAG_BAD_MOUSE if len(fields) > 1 else FLAG_NOT_ACTION_GRAMMAR
        )

    unknown: list[str] = []
    actions: list[list[str]] = []
    for f in fields[1:]:
        tokens = f.split()
        actions.append([n for n in tokens if n in KNOWN_NAMES][:MAX_KEYS])
        unknown.extend(n for n in tokens if n not in KNOWN_NAMES)
    if unknown:
        flags.append(FLAG_UNKNOWN_KEYS)
    if not actions:
        flags.append(FLAG_NO_ACTIONS)

    dx = dy = dz = 0
    if mouse_ok:
        dx, dy, dz = mouse
    idle = dx == 0 and dy == 0 and dz == 0 and not any(actions)
    if idle:
        flags.append(FLAG_IDLE)

    malformed = any(f in MALFORMED_FLAGS for f in flags)
    return {
        "flags": flags,
        "ok": not malformed,
        "malformed": malformed,
        "idle": idle,
        "unknown_keys": sorted(set(unknown))[:8],
        "action_count": len(actions),
        "text": stripped[:200],
    }


def validate(s: str, n_actions: int) -> None:
    """Raise when ``s`` does not match the training grammar exactly.

    ``n_actions`` is the expected chunk size — the number of actions the line must
    carry after the mouse field."""
    fields = [field.strip() for field in strip_wrappers(s).split(";")]
    if len(fields) != n_actions + 1:
        raise ValueError(f"expected {n_actions + 1} fields, got {len(fields)}")
    if _mouse_values(fields[0]) is None:
        raise ValueError(f"bad mouse field {fields[0]!r}")
    for field in fields[1:]:
        names = field.split()
        if len(names) > MAX_KEYS:
            raise ValueError(f"action names {len(names)} keys, max {MAX_KEYS}")
        unknown = [name for name in names if name not in KNOWN_NAMES]
        if unknown:
            raise ValueError(f"unknown key name(s) {unknown}")
