"""Roll a key-string checkpoint (or a floor) through OpenHA tasks in MineStudio.

    xvfb-run -a python benchmarks/openha/run.py --suite benchmarks/openha/suites/all801.json \
        --mode instructed --out runs/openha/LABEL [--base-url http://127.0.0.1:8000/v1] [--workers 8]

Modes: `blind` asks with the image alone; `instructed` adds the task's OpenHA description under
the brief system prompt; `noop` and `random` are the floors and need no server. A task succeeds
on its first reward, within its own `maximum_steps` ticks (600, 30 s at 20 Hz). Each episode
writes result.json beside MineStudio's recording; one already there is not run again. Score with
score.py.

Beyond scoring: --block-keys plays the policy with some keys taken out (an ablation), and
--relocate --save-frames rolls kill tasks out away from the benchmark's spawn spots and keeps
the frames.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import random
import signal
import time
from pathlib import Path

MODES = ("blind", "instructed", "noop", "random")


def family(task: str) -> str:
    return task.split(":")[0]


def relocate(cfg: dict, task: str, rep: int) -> dict:
    """A kill task moved off the one spot OpenHA spawns it at, for rollouts to train on that
    are not the benchmark's own episodes: a land task to the surface 600-2000 blocks away and
    at least 300 from every kill task's spot in that world, a water task 24-64 blocks along
    the same water. The mob is still summoned 2-7 blocks ahead of wherever the player lands."""
    import math
    from openagents.assets import KILL_ENTITY_SPAWN_FILE
    spawns = json.loads(Path(KILL_ENTITY_SPAWN_FILE).read_text())
    spot = spawns[task]["seeds"][0]
    x, y, z = spot["position"]
    taken = [s["position"] for v in spawns.values() for s in v["seeds"] if s["seed"] == spot["seed"]]
    rng = random.Random(f"relocate:{task}:{rep}")
    water = "water" in spawns[task]["label"]
    for _ in range(1000):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(24, 64) if water else rng.uniform(600, 2000)
        nx, nz = round(x + r * math.cos(a)), round(z + r * math.sin(a))
        if water or min(math.hypot(nx - p[0], nz - p[2]) for p in taken) >= 300:
            break
    cmds = list(cfg["callback"]["commands"])
    cmds[0] = f"/tp @s {nx} {y} {nz}" if water else f"/spreadplayers {nx} {nz} 0 16 false @s"
    return {**cfg, "callback": {**cfg["callback"], "commands": cmds}, "relocated": cmds[0]}


def write_video(path: Path, frames: list, texts: list) -> None:
    """The frames the policy was asked about, at the 5 Hz it was asked, each captioned with its answer."""
    import av
    import cv2
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=5, options={"crf": "23"})
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for img, text in zip(frames, texts):
            img = img.copy()
            cv2.putText(img, text.replace("<|action_start|>", "").replace("<|action_end|>", "")[:70],
                        (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def episode(job: dict) -> dict:
    import numpy as np
    from openagents.envs.env import env_init
    from openagents.envs.tasks.task_manager import choose_available_task
    import agent as ag

    task, rep, mode, out = job["task"], job["rep"], job["mode"], Path(job["out"])
    path = out / mode / task.replace(":", "_") / f"r{rep:02d}"
    if (path / "result.json").exists():
        return json.loads((path / "result.json").read_text())
    path.mkdir(parents=True, exist_ok=True)
    random.seed(rep)
    np.random.seed(rep)
    cfg = choose_available_task(task, difficulty=job["difficulty"])
    if job.get("relocate"):
        if family(task) != "kill_entity":
            raise ValueError(f"--relocate moves kill tasks only, not {task}")
        cfg = relocate(cfg, task, rep)
    pol = (ag.QwenKeyStringAgent(job["base_url"], temperature=job["temperature"],
                                 block_keys=job.get("block_keys") or (), history=job.get("history", 0),
                                 image_size=job.get("image_size"))
           if mode in ("blind", "instructed") else None)
    if pol:
        pol.reset(cfg["task_description"] if mode == "instructed" else None)
    t0 = time.time()
    env = env_init(cfg, str(path), argparse.Namespace(fps=20), obs_size=(640, 360))
    reset_s = time.time() - t0
    obs, reward, term, trunc, info = env.step(ag.noop())
    limit = int(job["max_steps"] or cfg["maximum_steps"])
    ticks, success, texts, guis, frames, jpegs = 0, False, [], [], [], []
    while ticks < limit and not success:
        gui = bool(info.get("is_gui_open"))
        guis.append(gui)
        frames.append(info["pov"])
        if pol:
            text, plan = pol.act(info["pov"], gui)
            jpegs.append(pol.last_jpeg)
        elif mode == "random":
            text = ""
            plan = [{k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in env.action_space.sample().items()} for _ in range(ag.TICKS)]
        else:
            text, plan = "", [ag.noop() for _ in range(ag.TICKS)]
        texts.append(text)
        for a in plan:
            obs, reward, term, trunc, info = env.step(a)
            ticks += 1
            if reward > 0:
                success = True
                break
            if ticks >= limit:
                break
    env.close()
    write_video(path / "pov.mp4", frames, texts)
    if job.get("save_frames") and jpegs:
        import tarfile
        with tarfile.open(path / "frames.tar", "w") as tar:
            for i, data in enumerate(jpegs):
                info_ = tarfile.TarInfo(f"{i:04d}.jpg")
                info_.size = len(data)
                tar.addfile(info_, io.BytesIO(data))
    res = {"task": task, "family": job.get("family") or family(task), "mode": mode, "rep": rep,
           "description": cfg["task_description"], "world_seed": cfg["seed"],
           "relocated": cfg.get("relocated"), "block_keys": sorted(job.get("block_keys") or ()),
           "temperature": job["temperature"] if pol else None, "history": job.get("history", 0),
           "image_size": job.get("image_size"),
           "success": success, "ticks": ticks, "decisions": len(texts), "gui_decisions": sum(guis),
           "gui": guis,
           "parse_failures": pol.failures if pol else 0,
           "latency_ms": round(1000 * pol.latency / max(1, pol.calls), 1) if pol else None,
           "reset_s": round(reset_s, 1), "wall_s": round(time.time() - t0, 1), "texts": texts}
    (path / "result.json").write_text(json.dumps(res, indent=1))
    return res


class EpisodeTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise EpisodeTimeout("over --timeout")


def attempt(job: dict) -> dict:
    """episode(), tried again when it raises or overruns. With a dozen simulators starting at once
    MineStudio's reset now and then times out, or never returns while Java spins on 15 cores; one
    such episode would otherwise end the pool or leave it waiting forever. The simulator of a failed
    try is killed with it. A job that fails every try writes no result.json, so the next run of the
    suite picks it up."""
    import psutil
    signal.signal(signal.SIGALRM, _alarm)
    for i in range(job["tries"]):
        signal.alarm(job["timeout"])
        try:
            return episode(job)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            print(f"  {job['task']} r{job['rep']}: try {i + 1} raised {error}", flush=True)
            for child in psutil.Process().children(recursive=True):
                child.kill()
        finally:
            signal.alarm(0)
    return {"task": job["task"], "rep": job["rep"], "error": error}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", help="json: {tasks: [...], repeats: N, families: {task: family}}; "
                                    "families is optional, for tasks whose name does not say it "
                                    "(OpenHA names smelting craft_item:<output>)")
    ap.add_argument("--tasks", help="comma-separated task names, instead of --suite")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--difficulty", default="zero")
    ap.add_argument("--max-steps", type=int, help="ticks per episode; default the task's own")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling, as OpenHA's agents use")
    ap.add_argument("--tries", type=int, default=3, help="per episode, when the simulator fails")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds per try; a normal episode takes 1-5 min with its reset")
    ap.add_argument("--block-keys", default="",
                    help="comma-separated keys the policy may generate but that are not played")
    ap.add_argument("--save-frames", action="store_true",
                    help="keep every JPEG the policy was asked about (frames.tar)")
    ap.add_argument("--relocate", action="store_true",
                    help="kill tasks away from OpenHA's spawn spots (relocate())")
    ap.add_argument("--history", type=int, default=0,
                    help="earlier decisions sent before each frame (agent.py plays 0 only)")
    ap.add_argument("--image-size", help="WIDTHxHEIGHT each frame is resized to (agent.py sends frames at "
                                         "their own size only)")
    args = ap.parse_args()
    image_size = [int(v) for v in args.image_size.lower().split("x")] if args.image_size else None
    suite = json.loads(Path(args.suite).read_text()) if args.suite else {}
    tasks = args.tasks.split(",") if args.tasks else suite["tasks"]
    repeats = args.repeats or suite.get("repeats", 1)
    families = suite.get("families", {})
    jobs = [{"task": t, "rep": r, "mode": args.mode, "out": str(Path(args.out).expanduser()),
             "base_url": args.base_url, "difficulty": args.difficulty, "max_steps": args.max_steps,
             "temperature": args.temperature, "family": families.get(t), "tries": args.tries,
             "timeout": args.timeout, "save_frames": args.save_frames, "relocate": args.relocate,
             "block_keys": [k for k in args.block_keys.split(",") if k], "history": args.history,
             "image_size": image_size}
            for t in tasks for r in range(repeats)]
    print(f"{len(jobs)} episodes ({len(tasks)} tasks x {repeats}), mode {args.mode}", flush=True)
    errors = 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(attempt, jobs):
            if "error" in res:
                errors += 1
                print(f"  {res['task']} r{res['rep']}: not run, {res['error']}", flush=True)
                continue
            print(f"  {res['task']} r{res['rep']}: {'success' if res['success'] else 'fail'} "
                  f"at {res['ticks']} ticks, {res['parse_failures']}/{res['decisions']} unparsed, "
                  f"{res['wall_s']}s", flush=True)
    if errors:
        raise SystemExit(f"{errors} of {len(jobs)} episodes did not run; run the suite again for them")


if __name__ == "__main__":
    main()
