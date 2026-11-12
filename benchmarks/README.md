# Evaluation setup

How to evaluate Script2Play-4B on the paper's three benchmarks, with the settings the paper used.
All three query the model through a vLLM server:

```bash
pip install -e .
bash scripts/download_checkpoint.sh                        # checkpoints/script2play-4b
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh checkpoints/script2play-4b 8000
```

The commands below assume the server at `http://127.0.0.1:8000/v1` and the weights at
`checkpoints/script2play-4b`. Sampling is at temperature 0.8, so two runs differ: expect a map's
50-episode ViZDoom mean to move by up to ±0.5 to ±2 reward between runs, and OpenHA all801 by
about ±2 points. Run one benchmark at a time on a machine that is otherwise idle; OpenHA in
particular slows down and scores lower when MineStudio shares the CPUs.

## ViZDoom

GameWAM's four-map protocol: battle1, battle2, defend_the_line and defend_the_center, 50 episodes
each, at most 2100 tics at 35 Hz; the score is a map's mean episode reward (kills; Defend Line and
Defend Center take a point off when the player dies).

1. Clone [GameWAM](https://github.com/yunncheng/GameWAM) at commit `e3a3d29` and set up its
   environment (ViZDoom and its Python dependencies) as its README describes.
2. In that environment, install this repo: `pip install -e /path/to/script2play`.
3. Serve the model (above) and run:

```bash
python benchmarks/vizdoom/eval_gamewam.py --gamewam /path/to/GameWAM \
    --checkpoint checkpoints/script2play-4b --output-dir runs/vizdoom_script2play_4b --workers 8 --no-video
cat runs/vizdoom_script2play_4b/summary.txt
```

GameWAM's own episode loop runs the episodes (its maps, seeds, observation and reward
bookkeeping); `eval_gamewam.py` only supplies the policy. Each call sends GameWAM's task text as
`"Control the Doom player to complete the task: {task}"` with the 640x480 frame resized to 832 px
wide, and plays the reply over 7 tics: `LB` shoots, `W`/`Up` and `S`/`Down` move, `A`/`D`
strafe, `Shift`/`Ctrl` speed up, and `dx` turns the view at 0.15 deg/px (plus 120 deg/s for
`Left`/`Right`). The run writes `summary.txt` (mean reward per map) and each episode's
`results.json`, and resumes where it stopped if started again with the same arguments. 8 workers
finish the 200 episodes in about 20 minutes.

## OpenHA

Minecraft instruction following on [OpenHA](https://github.com/CraftJarvis/OpenHA)'s tasks in
MineStudio: all 801 task definitions (Embodied: mining and block interaction; GUI: crafting and
smelting; Combat: killing a mob), 2 episodes each, with the task's description as the
instruction. An episode succeeds on the task's first reward within its 600 ticks (30 s at 20 Hz);
`eval255` is the 255-task subset (all Embodied and Combat tasks, 40 GUI tasks).

1. A Python 3.10 environment with the pinned OpenHA stack (MineStudio 1.1.6, OpenHA `606efc6`):
   `pip install -r benchmarks/openha/requirements-openha.txt`, then `pip install -e .`.
2. Java 8 (OpenJDK 1.8, for MineStudio's Minecraft) and `xvfb-run`.
3. MineStudio's engine (445 MB, from Hugging Face's CraftJarvis/SimulatorEngine), downloaded into
   the run's `MINESTUDIO_DIR` before the run: the simulator only asks for it interactively,
   which run.py's workers cannot answer. Give each concurrent run its own `MINESTUDIO_DIR`.
4. Serve the model (above) and run:

```bash
export MINESTUDIO_DIR=/tmp/ms_script2play_4b
python -c "from minestudio.simulator.entry import download_engine; download_engine()"
export SCRIPT2PLAY_CHECKPOINT=checkpoints/script2play-4b   # the agent reads its tokenizer.json
xvfb-run -a python benchmarks/openha/run.py \
    --suite benchmarks/openha/suites/all801.json --mode instructed --out runs/openha/script2play_4b \
    --base-url http://127.0.0.1:8000/v1 --workers 8
python benchmarks/openha/score.py runs/openha/script2play_4b
python benchmarks/openha/score.py runs/openha/script2play_4b --suite benchmarks/openha/suites/eval255.json
```

`agent.py` sends the 640x360 frame at its own size (not resized), with the `minecraft` prompt
label and MineStudio's task description as it is; a reply that is not exactly the 4-step grammar
plays as idle and is counted in each episode's `parse_failures`. The mouse turns the camera at
0.15 deg/px in the world and moves the cursor at 0.144/0.145 deg/px (x/y) when a screen is open.
`score.py` reports the success rate by group (Embodied, GUI, Combat) and by task. The 1602
episodes take about 3.5 hours with 8 workers. Each episode writes a `result.json` and is not run
again, so the same command resumes a run or retries the episodes that failed.

## OmniGameArena

[OmniGameArena](https://arxiv.org/abs/2606.09826)'s five 3D solo games, 5 episodes each, scored
with the benchmark's per-game normalized metrics and their mean. This repo does not include an
OmniGameArena agent; connect the model to the benchmark as follows.

1. Install OmniGameArena and its UE5 game builds as its documentation describes (rendering needs
   a GPU).
2. Run it in its time-paused clock mode, so the game waits while the model decides and model
   latency does not change the scores.
3. Give the benchmark an agent that, every 200 ms of game time, sends the current screenshot to
   the model with no task line (the games show their task on screen) and plays the reply as four
   50 ms steps:

| Game | Prompt label (`game=`) | Keys the game takes | Remap |
|---|---|---|---|
| `obstacle_run_3d` | `obstacle-run-3d` | W A S D Space | |
| `last_stand` | `last-stand` | W A S D Space | |
| `monster_shoot` | `monster-shoot` | W A S D Space LB | |
| `scene_escape` | `scene-escape` | W A S D F Space | RB plays as F (interact) |
| `cue_chase` | `cue-chase` | W A S D F Space | RB plays as F (interact) |

```python
from script2play import MOUSE_DEG_PER_PX, VLAClient

GAMES = {  # OmniGameArena game: (prompt label, keys the game takes, remap)
    "obstacle_run_3d": ("obstacle-run-3d", {"W", "A", "S", "D", "Space"}, {}),
    "last_stand": ("last-stand", {"W", "A", "S", "D", "Space"}, {}),
    "monster_shoot": ("monster-shoot", {"W", "A", "S", "D", "Space", "LB"}, {}),
    "scene_escape": ("scene-escape", {"W", "A", "S", "D", "F", "Space"}, {"RB": "F"}),
    "cue_chase": ("cue-chase", {"W", "A", "S", "D", "F", "Space"}, {"RB": "F"}),
}

label, keys, remap = GAMES[game]
client = VLAClient("http://127.0.0.1:8000/v1", "checkpoints/script2play-4b", game=label)
held = []
while not done:                                       # the game paused while the model decides
    chunk = client.act(screenshot, "")                # resized to 832 px wide; no task line
    for step in chunk.steps(held_before=held):        # 4 steps of 50 ms
        play = {remap.get(k, k) for k in step.held} & keys    # LB is the left mouse button
        yaw = step.dx * MOUSE_DEG_PER_PX + 120 * 0.05 * (("Right" in step.held) - ("Left" in step.held))
        pitch = step.dy * MOUSE_DEG_PER_PX
        ...  # hold `play` for 50 ms and turn the camera by (yaw, pitch) degrees
        held = step.held
```

The client's defaults are the rest of the paper's settings: the brief system prompt, the frame
as JPEG quality 88, no history, no `<|action_start|>` prefill, temperature 0.8, top-p 0.99, at
most 48 tokens. Keys a game does not take are generated but not played.
