# Script2Play inference

Inference for **Script2Play-4B**, the Qwen3.5-4B vision-language-action model of *From Game
Generation to Game Playing: Fully Automated Ability Training for VLA Models*
([paper](https://arxiv.org/abs/2611.11136), [project page](https://assettool-labs.github.io/script2play/)).
It plays games from the screen: one frame and a task in, the next 200 ms of keyboard and mouse out.

The client sends exactly the request the paper's evaluations used, and
[benchmarks/](benchmarks/README.md) describes how to set up ViZDoom, OpenHA and OmniGameArena
with it.

## Quick start

```bash
pip install -e .                       # the client: requests, pillow, numpy
pip install vllm==0.26.0               # the server (served with vLLM 0.26.0, transformers 5.20.1, torch 2.11)

bash scripts/download_checkpoint.sh    # 8.5 GB into checkpoints/script2play-4b
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh checkpoints/script2play-4b 8000

python -m script2play --base-url http://127.0.0.1:8000/v1 \
    --checkpoint checkpoints/script2play-4b --image frame.png \
    --task "Defend the center: shoot the enemies closing in from all directions."
```

The weights are [GameIntelligenceLab/Script2Play-4B](https://huggingface.co/GameIntelligenceLab/Script2Play-4B)
on Hugging Face, private for now: `hf auth login` with an account that has access first.

From Python:

```python
from script2play import DOOM_TASK, VLAClient

client = VLAClient("http://127.0.0.1:8000/v1", "checkpoints/script2play-4b", game="doom")
chunk = client.act(frame, DOOM_TASK.format(task="Defend the center: shoot the enemies."), seed=0)
chunk.dx, chunk.dy   # mouse px over the 200 ms, e.g. -150, 0
chunk.turn_deg       # the view turn that is: (-22.5, 0.0) degrees
chunk.keys           # keys held in each 50 ms step, e.g. [["LB"], ["LB"], [], []]

held = []
for step in chunk.steps(held_before=held):   # 4 steps, t_ms 0, 50, 100, 150
    ...  # press step.pressed, release step.released, move the mouse by step.dx, step.dy
    held = step.held
```

`frame` is a PIL image, an HxWx3 uint8 array or a path; it is resized to 832 px wide and sent as
a JPEG at quality 88. `steps()` splits the mouse px evenly over the 4 steps (summing to `dx`,
`dy`) and gives each step's key presses and releases against the step before.

A request that fails 3 times (`retries`) returns an idle chunk with `chunk.error` set instead of
raising, so a control loop keeps running.

## What the model outputs

One line per call, e.g. `-150 0 0 ; LB ; LB W ; ; Left`:

- `dx dy dz`: the mouse displacement in px summed over the 200 ms (x right, y down) and the scroll
  steps. **One px is 0.15 degrees of view turn**, the gain the training games were recorded at and
  the one every score below used (`MOUSE_DEG_PER_PX`). Spread it evenly over the chunk.
- Then 4 groups of keys, one per 50 ms step, held for that step: a key in consecutive groups is
  held through them. Names: letters `A`-`Z`, `Space`, `Shift`, `Ctrl`, `Tab`, `Esc`, `Enter`,
  arrows `Up` `Down` `Left` `Right`, digits as words (`one`), mouse `LB` `RB` `MB`.

`parse` never raises: a malformed reply becomes an idle chunk, and `chunk.flags` /
`chunk.malformed` say why (prose instead of the action grammar, a bad mouse field, unknown keys,
or a failed request).

In the ViZDoom evaluation a chunk was played over 7 tics at 35 tics/s (groups on tics 0-1, 2-3,
4-5, 6), with `LB` shoot, `W`/`Up` forward, `S`/`Down` back, `A`/`D` strafe, `Shift`/`Ctrl`
speed, and `dx * 0.15 / 7` degrees of turn a tic (plus 120 deg/s for `Left`/`Right`), capped at
10 degrees a tic.

## Prompt and sampling

- System prompt: the brief instruction prompt the model was trained on (`script2play.prompt`).
  `game="doom"` for Doom-like games: the task goes into `DOOM_TASK`, `"Control the Doom player
  to complete the task: {task}"`, capitalised and closed with a full stop (`as_instruction`).
- Minecraft is different, as its data was: frames at their own 640x360 and MineStudio's task
  description as it is, `VLAClient(..., game="minecraft", resize_width=None, normalize_task=False)`
  (what `benchmarks/openha/agent.py` does).
- The user turn is the frame alone; no history; no thinking (`enable_thinking: false`).
- Temperature 0.8, top-p 0.99, no top-k, at most 48 tokens, stopping at `<|action_end|>` or
  `<|im_end|>` (ids from the checkpoint's `tokenizer.json`), special tokens skipped.
- One call every 200 ms of game time; pass a `seed` to make a call reproducible.

## The model

Qwen3.5-4B-Base post-trained by behavior cloning on 150.3 h of verified scripted-teacher
demonstrations: 26.9 h of Minecraft and 123.4 h of generated Roblox games. No ViZDoom or UE5
gameplay is in the training data, and no human gameplay, RL or DAgger is used.

## Evaluation

[benchmarks/README.md](benchmarks/README.md) gives the setup for each benchmark, run as in the
paper:

- **ViZDoom**: GameWAM's four maps (battle1, battle2, defend_the_line, defend_the_center), 50
  episodes each, with `benchmarks/vizdoom/eval_gamewam.py` on a GameWAM checkout.
- **OpenHA**: Minecraft instruction following (all801 and its eval255 subset, 2 episodes a task)
  in MineStudio, with the runner, scorer and suites in `benchmarks/openha/`.
- **OmniGameArena**: five UE5 solo games under the time-paused protocol, with the per-game prompt
  labels, key sets and mouse mapping to give the benchmark's agent.

## Layout

- `script2play/client.py`: frame encoding, the request, the reply parsed into an `ActionChunk`
  and its timed `Step`s; `prompt.py` and `actions.py`: the training prompt and action grammar.
- `scripts/serve.sh`, `scripts/download_checkpoint.sh`; `tests/` (`pytest`).
- `benchmarks/`: `vizdoom/eval_gamewam.py`; `openha/` (OpenHA runner, scorer and task suites, with
  `agent.py` on this client); the setup of all three benchmarks in `benchmarks/README.md`.
