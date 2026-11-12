"""A served checkpoint on GameWAM's four-map ViZDoom benchmark (battle1, battle2, defend_the_line,
defend_the_center; 50 episodes a map), through script2play's client.

    python benchmarks/vizdoom/eval_gamewam.py --gamewam ~/GameWAM --base-url http://127.0.0.1:8000/v1 \
        --checkpoint checkpoints/script2play-4b --output-dir runs/vizdoom_script2play_4b --workers 8

Episodes run through GameWAM's own loop (github.com/yunncheng/GameWAM e3a3d29,
experiments/vizdoom/eval_vizdoom_multi.py: ``EpisodeWorker.run_episode``), so maps, per-episode
seeds, the 2100-tic cap, the 640x480 RGB observation, reward bookkeeping and the summary/CSV
reports are theirs; only the policy is ours. This is the harness the reported ViZDoom scores were
measured with.

One call plans a 200 ms chunk of 4 groups of 50 ms, played over 7 tics at 35 tics/s (groups on
tics 0-1, 2-3, 4-5, 6). Keys map onto GameWAM's 9-dim action: LB -> shoot, W/Up -> forward,
S/Down -> backward, A -> strafe left, D -> strafe right, Shift/Ctrl -> speed; other keys are
dropped and counted. The mouse's dx (0.15 deg/px, spread evenly over the chunk) plus Left/Right
(120 deg/s) go to ``turn_delta`` (+1 = 10 deg/tic to the right); dy has no ViZDoom button. The
task line is GameWAM's prompt ("Control the Doom player to complete the task: {task}"), and each
call's sampling seed is the episode's model seed plus the calls made so far in it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script2play import DOOM_TASK, VLAClient, actions  # noqa: E402

logger = logging.getLogger("vizdoom_eval")

CHUNK_GROUPS = 4
CHUNK_TICS = 7
TICS_PER_SECOND = 35.0
TURN_DELTA_MAX_DEG = 10.0
ARROW_YAW_DEG_PER_S = 120.0
MOUSE_DEG_PER_PX = 0.15
KEY_TO_ACTION = {"LB": 0, "W": 1, "Up": 1, "S": 2, "Down": 2, "A": 3, "D": 4, "Shift": 5, "Ctrl": 5}
ARROW_YAW = {"Left": -1.0, "Right": 1.0}


def _gamewam_eval(gamewam: Path):
    for p in (gamewam, gamewam / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(
        "gamewam_eval_vizdoom_multi", gamewam / "experiments" / "vizdoom" / "eval_vizdoom_multi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DoomPolicy:
    """GameWAM's inference stand-in: ``get_action`` returns one 9-dim action per tic."""

    def __init__(self, *, base_url: str, checkpoint_path: str, temperature: float = 0.8, top_p: float = 0.99,
                 top_k: int = 0, timeout: float = 30.0):
        self.client = VLAClient(base_url, checkpoint_path, game="doom", temperature=temperature, top_p=top_p,
                                top_k=top_k, timeout=timeout)
        self.seed = 0
        self.inference_logger = None
        self.reset()

    def reset(self) -> None:
        self._plan: list[np.ndarray] = []
        self._calls = 0
        self._tic = 0
        self.stats = {"calls": 0, "errors": 0, "malformed": 0, "dropped_keys": {}}

    def get_action(self, obs, instruction: str) -> np.ndarray:
        if not self._plan:
            self._plan = self._replan(obs.image, instruction)
        self._tic += 1
        return self._plan.pop(0)

    def _replan(self, image, task: str) -> list[np.ndarray]:
        line = DOOM_TASK.format(task=task)
        sampling_seed = (int(self.seed) + self._calls) & 0x7FFFFFFF
        t0 = time.perf_counter()
        chunk = self.client.act(image, line, seed=sampling_seed)
        latency = time.perf_counter() - t0
        self._calls += 1
        self.stats["calls"] += 1
        if chunk.error is not None:
            self.stats["errors"] += 1
            logger.warning("request failed: %s", chunk.error)
        elif chunk.malformed:
            self.stats["malformed"] += 1
        plan, debug = self.translate(chunk)
        debug.update({"response": None if chunk.error else
                      f"{actions.ACTION_START}{actions.strip_wrappers(chunk.text)}{actions.ACTION_END}",
                      "error": chunk.error})
        if self.inference_logger is not None:
            self.inference_logger({
                "step_count": self._tic,
                "prompt": self.client.system_prompt(line),
                "task_str": task,
                "sampling_seed": sampling_seed,
                "base_seed": self.seed,
                "global_replan_index": self._calls - 1,
                "prediction_horizon": CHUNK_TICS,
                "execution_horizon": CHUNK_TICS,
                "infer_latency_sec": latency,
                "action_chunk": np.stack(plan),
                "action_debug": debug,
            })
        return plan

    def translate(self, chunk) -> tuple[list[np.ndarray], dict]:
        mouse_yaw_per_tic = chunk.dx * MOUSE_DEG_PER_PX / CHUNK_TICS
        arrow_yaw_per_tic = ARROW_YAW_DEG_PER_S / TICS_PER_SECOND
        dropped: set[str] = set()
        plan = []
        for tic in range(CHUNK_TICS):
            group = chunk.keys[min(CHUNK_GROUPS - 1, tic * CHUNK_GROUPS // CHUNK_TICS)]
            vec = np.zeros(9, dtype=np.float32)
            yaw = mouse_yaw_per_tic
            for name in group:
                if name in ARROW_YAW:
                    yaw += ARROW_YAW[name] * arrow_yaw_per_tic
                elif name in KEY_TO_ACTION:
                    vec[KEY_TO_ACTION[name]] = 1.0
                else:
                    dropped.add(name)
            vec[8] = float(np.clip(yaw / TURN_DELTA_MAX_DEG, -1.0, 1.0))
            plan.append(vec)
        for name in dropped:
            self.stats["dropped_keys"][name] = self.stats["dropped_keys"].get(name, 0) + 1
        return plan, {"dx": chunk.dx, "dy": chunk.dy, "groups": chunk.keys,
                      "dropped_keys": sorted(dropped), "flags": chunk.flags}


_WORKER = None


def _init_worker(gamewam: str, policy_kwargs: dict, worker_kwargs: dict) -> None:
    global _WORKER
    logging.basicConfig(level=logging.WARNING)
    gw = _gamewam_eval(Path(gamewam))
    worker = object.__new__(gw.EpisodeWorker)
    worker.inference = DoomPolicy(**policy_kwargs)
    worker.ckpt = policy_kwargs["checkpoint_path"]
    worker.dataset_stats_path = None
    worker.device = "remote"
    worker.max_steps = int(worker_kwargs["max_steps"])
    worker.save_video = bool(worker_kwargs["save_video"])
    worker.prompt_index = int(worker_kwargs["prompt_index"])
    worker.window_visible = False
    worker.fps = int(worker_kwargs["fps"])
    worker.base_seed = int(worker_kwargs["seed"])
    worker.eval_cfg = worker_kwargs["eval_cfg"]
    _WORKER = worker


def _run_episode(scenario: str, episode_id: str, episode_dir: str) -> dict:
    result = _WORKER.run_episode(scenario, episode_id, episode_dir)
    stats = _WORKER.inference.stats
    result["policy_stats"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in stats.items()}
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gamewam", required=True, help="a checkout of github.com/yunncheng/GameWAM (e3a3d29)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--checkpoint", required=True, help="the served checkpoint's directory (its tokenizer.json)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--episodes-per-map", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2100)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.99)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    gamewam = Path(args.gamewam).expanduser().resolve()
    gw = _gamewam_eval(gamewam)
    policy_kwargs = {"base_url": args.base_url, "checkpoint_path": str(Path(args.checkpoint).resolve()),
                     "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    scenarios = gw._resolve_scenarios(args.scenarios)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"policy": policy_kwargs, "scenarios": scenarios, "episodes_per_map": args.episodes_per_map,
            "max_steps": args.max_steps, "base_seed": args.seed, "prompt_index": args.prompt_index,
            "chunk_tics": CHUNK_TICS}
    gw._validate_resume_meta(output_dir / "run_meta.json", json.loads(json.dumps(meta)))

    results_by_scenario, pending = {}, []
    for scenario in scenarios:
        results = gw._load_json(output_dir / scenario / "results.json", [])
        results.sort(key=lambda r: int(r["episode_id"]))
        done = {str(r["episode_id"]) for r in results}
        results_by_scenario[scenario] = results
        pending += [(scenario, str(i)) for i in range(args.episodes_per_map) if str(i) not in done]
    gw._write_reports(output_dir, scenarios, results_by_scenario, args.episodes_per_map)
    logger.info("%s: %d episodes pending, %d workers", output_dir.name, len(pending), args.workers)
    if not pending:
        return

    worker_kwargs = {"max_steps": args.max_steps, "save_video": not args.no_video,
                     "prompt_index": args.prompt_index, "fps": args.fps, "seed": args.seed,
                     "eval_cfg": {"policy": policy_kwargs, "chunk_tics": CHUNK_TICS, "prompt_template": DOOM_TASK}}
    failures = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(gamewam), policy_kwargs, worker_kwargs)) as pool:
        futures = {}
        for scenario, episode_id in pending:
            ep_dir = output_dir / scenario / episode_id
            ep_dir.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(_run_episode, scenario, episode_id, str(ep_dir))] = (scenario, episode_id)
        while futures:
            finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in finished:
                scenario, episode_id = futures.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    failures += 1
                    logger.error("[%s] ep=%s failed: %s", scenario, episode_id, exc)
                    continue
                results = results_by_scenario[scenario]
                results.append(result)
                results.sort(key=lambda r: int(r["episode_id"]))
                gw._dump_json(results, output_dir / scenario / "results.json")
                summary = gw._write_reports(output_dir, scenarios, results_by_scenario, args.episodes_per_map)
                item = summary["maps"][scenario]
                logger.info("[%s] ep=%s reward=%.2f steps=%d calls=%d; average=%.3f (%d/%d)",
                            scenario, episode_id, result["episode_reward"], result["steps"],
                            result["policy_stats"]["calls"], item["average_episode_reward"],
                            item["episodes_completed"], item["episodes_target"])
    logger.info("done: %s (%d failed episodes)", output_dir, failures)


if __name__ == "__main__":
    main()
