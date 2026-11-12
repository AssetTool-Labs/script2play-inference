# Reproducing the benchmark scores

The three benchmarks Script2Play-4B was scored on, each run the way the reported numbers were
measured. Serve the checkpoint first (`scripts/download_checkpoint.sh`, then `scripts/serve.sh`);
every command below assumes it at `http://127.0.0.1:8000/v1` and the weights at
`checkpoints/script2play-4b`. Sampling is at temperature 0.8, so a rerun lands near the reported
scores, not on them: the 95% intervals of the reported scores are ±1.5 (battle1), ±1.0 (battle2),
±1.9 (defend_the_line) and ±0.5 (defend_the_center) on ViZDoom, and ±2.3 points on OpenHA all801.

## ViZDoom (GameWAM's 4 maps x 50 episodes)

Needs a checkout of [GameWAM](https://github.com/yunncheng/GameWAM) at e3a3d29 in an
environment with its ViZDoom dependencies, plus `requests`.

```bash
python benchmarks/vizdoom/eval_gamewam.py --gamewam /path/to/GameWAM \
    --checkpoint checkpoints/script2play-4b --output-dir runs/vizdoom_script2play_4b --workers 8 --no-video
cat runs/vizdoom_script2play_4b/summary.txt
```

Reported (mean reward): battle1 17.00, battle2 4.08, defend_the_line 22.94, defend_the_center
7.66 (mean 12.92). A rerun from a fresh clone gave 17.22, 3.66, 24.24, 7.52 (mean 13.16). A run
resumes where it stopped.

## OpenHA (all801 and eval255, instructed, 2 episodes a task)

Needs MineStudio 1.1.6 and [OpenHA](https://github.com/CraftJarvis/OpenHA) at 606efc6
(`pip install -r benchmarks/openha/requirements-openha.txt`), Java for MineStudio's engine and
`xvfb-run`. `agent.py` plays the checkpoint through this repo's client. Download MineStudio's
engine (445 MB, from Hugging Face's CraftJarvis/SimulatorEngine) into the run's `MINESTUDIO_DIR`
first: the simulator only asks for it interactively, which run.py's workers cannot answer. Give
each concurrent run its own `MINESTUDIO_DIR`.

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

Reported (success): all801 66.7% (Embodied 63.9, GUI 66.4, Combat 78.6), eval255 67.5%. A rerun
from a fresh clone gave 68.9% and 69.4%. 1602 episodes take about 3.5 hours with 8 workers.
Episodes already written are not run again.

## OmniGameArena (5 solo games x 5 episodes)

The OmniGameArena adapter is not included. The reported scores (ObstacleRun3D 0.374, LastStand
0.436, MonsterShoot 0.972, SceneEscape 0.420, CueChase 1.000, mean 0.640) used the
time-paused protocol and these settings, the same request as this client's defaults:

| Setting | Value |
|---|---|
| Games | obstacle_run_3d, last_stand, monster_shoot, scene_escape, cue_chase; 5 episodes each |
| Prompt | the brief system prompt (`script2play.prompt`) |
| Frame | resized to 832 px wide, JPEG quality 88; the current frame only (no history) |
| Mouse | 0.15 degrees of view turn per px of `dx`/`dy` |
| Sampling | temperature 0.8, top-p 0.99, no top-k, at most 48 tokens |
| Action prefix | none (the reply starts the action line itself) |
| Interact | the model's right mouse button (`RB`) played as `F` (interact) in SceneEscape and CueChase; the other games as they are |
| Cadence | one call per 200 ms of game time, its 4 key groups played 50 ms each |
