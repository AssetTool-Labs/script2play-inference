"""One call to a served checkpoint: a game frame and a task in, a 200 ms action chunk out.

The request is the one the reported scores were measured with: the brief system prompt with the
task line, the frame alone in the user turn (832 px wide, JPEG 88), no thinking, sampling stopped
at ``<|action_end|>`` or ``<|im_end|>``, special tokens skipped.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from . import actions, prompt

RESIZE_WIDTH = 832
JPEG_QUALITY = 88
MAX_TOKENS = 48
CHUNK_ACTIONS = 4
ACTION_MS = 50
# View degrees one px of the model's dx/dy turns: the gain the training games were recorded at,
# and the one every reported score used.
MOUSE_DEG_PER_PX = 0.15
DOOM_TASK = "Control the Doom player to complete the task: {task}"


@dataclass
class Step:
    """One 50 ms step of a chunk: the keys held through it, those pressed and released at its
    start (against the step before; the chunk's first step against ``held_before``), and its
    share of the chunk's mouse px."""

    t_ms: int
    held: list[str]
    pressed: list[str]
    released: list[str]
    dx: int
    dy: int


@dataclass
class ActionChunk:
    """dx, dy: mouse px summed over the 200 ms (x right, y down); dz: scroll steps; keys: the keys
    held in each 50 ms step, in order (a key in consecutive steps is held through them). error:
    why the request failed, when it did (the chunk is then idle)."""

    dx: int
    dy: int
    dz: int
    keys: list[list[str]]
    text: str
    flags: list[str]
    error: str | None = None

    @property
    def malformed(self) -> bool:
        return self.error is not None or any(f in actions.MALFORMED_FLAGS for f in self.flags)

    @property
    def turn_deg(self) -> tuple[float, float]:
        """The view turn the chunk asks for, (yaw right, pitch down) in degrees."""
        return self.dx * MOUSE_DEG_PER_PX, self.dy * MOUSE_DEG_PER_PX

    def steps(self, held_before: list[str] | None = None) -> list[Step]:
        """The chunk as CHUNK_ACTIONS timed steps, the mouse px split evenly over them (the first
        steps take the remainder, so the steps sum to dx and dy)."""
        prev = list(held_before or [])
        out = []
        for k, keys in enumerate(self.keys):
            out.append(Step(k * ACTION_MS, list(keys), [n for n in keys if n not in prev],
                            [n for n in prev if n not in keys], _share(self.dx, k), _share(self.dy, k)))
            prev = keys
        return out


def _share(total: int, k: int) -> int:
    q, r = divmod(abs(total), CHUNK_ACTIONS)
    return (q + (k < r)) * (1 if total >= 0 else -1)


def stop_token_ids(checkpoint_dir: str | Path) -> list[int]:
    tok = json.loads((Path(checkpoint_dir) / "tokenizer.json").read_text())
    added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    return [int(added[n]) for n in (actions.ACTION_END, "<|im_end|>")]


def as_instruction(text: str) -> str:
    """The task line as training wrote it: a capital first letter and closing punctuation."""
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def encode_jpeg(image, width: int | None = RESIZE_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    """A frame (PIL image, HxWx3 uint8 array or image path) as the JPEG the model reads: scaled
    bicubic to ``width`` px wide, or sent at its own size with ``width=None``."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    img = img.convert("RGB")
    if width and img.width != width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def encode_frame(image, width: int | None = RESIZE_WIDTH, quality: int = JPEG_QUALITY) -> str:
    """encode_jpeg's JPEG as a data URL."""
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(image, width, quality)).decode()


def parse(text: str) -> ActionChunk:
    """The model's reply as an action chunk, padded or cut to CHUNK_ACTIONS steps."""
    decoded = actions.decode(text)
    keys = [list(k) for k in decoded["actions"][:CHUNK_ACTIONS]]
    keys += [[] for _ in range(CHUNK_ACTIONS - len(keys))]
    return ActionChunk(decoded["dx"], decoded["dy"], decoded["dz"], keys, text,
                       actions.diagnose(text)["flags"])


class VLAClient:
    """A served checkpoint behind vLLM's OpenAI-compatible endpoint (scripts/serve.sh).

    ``game`` is the prompt's game label (see prompt.instruction_system): "doom" for Doom-like
    games, "minecraft" for Minecraft; any other label is called "Roblox <label>" with the default
    controls.

    The settings differ by game as the data did, and as each benchmark was run (benchmarks/):
    Doom-like and Roblox frames go 832 px wide with the task line capitalised and closed
    (``resize_width=832, normalize_task=True``, the defaults); Minecraft frames go at their own
    640x360 with MineStudio's task description as it is (``game="minecraft", resize_width=None,
    normalize_task=False``)."""

    def __init__(self, base_url: str, checkpoint_dir: str | Path, *, game: str = "doom",
                 brief: bool = True, temperature: float = 0.8, top_p: float = 0.99, top_k: int = 0,
                 max_tokens: int = MAX_TOKENS, timeout: float = 30.0, resize_width: int | None = RESIZE_WIDTH,
                 normalize_task: bool = True):
        self.base_url = base_url.rstrip("/")
        self.game = game
        self.brief = brief
        self.resize_width = resize_width
        self.normalize_task = normalize_task
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.stop_token_ids = stop_token_ids(checkpoint_dir)
        self.session = requests.Session()
        self.model = self.session.get(f"{self.base_url}/models", timeout=self.timeout).json()["data"][0]["id"]

    def system_prompt(self, instruction: str) -> str:
        task = as_instruction(instruction) if self.normalize_task else instruction
        return prompt.instruction_system(self.game, task, brief=self.brief)

    def request(self, image, instruction: str, seed: int | None = None) -> dict:
        """The chat-completions body; ``image`` may also be JPEG bytes already encoded."""
        jpeg = image if isinstance(image, bytes) else encode_jpeg(image, self.resize_width)
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt(instruction)},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "skip_special_tokens": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stop_token_ids": self.stop_token_ids,
        }
        if self.temperature > 0 and self.top_k > 0:
            body["top_k"] = self.top_k
        if seed is not None:
            body["seed"] = int(seed) & 0x7FFFFFFF
        return body

    def act(self, image, instruction: str, seed: int | None = None, retries: int = 3) -> ActionChunk:
        """One chunk for the frame. A request that fails ``retries`` times gives an idle chunk with
        ``error`` set, as the evaluation harness did, so a control loop keeps its clock."""
        body = self.request(image, instruction, seed)
        error = None
        for attempt in range(retries):
            try:
                resp = self.session.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
                resp.raise_for_status()
                return parse(resp.json()["choices"][0]["message"]["content"] or "")
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                error = str(exc)
                if attempt + 1 < retries:
                    time.sleep(1.0 + attempt)
        chunk = parse("")
        chunk.error = error
        return chunk
