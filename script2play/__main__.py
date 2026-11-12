"""Ask a served checkpoint for one action chunk on an image.
    python -m script2play --base-url http://127.0.0.1:8000/v1 --checkpoint checkpoints/script2play-4b \
        --image frame.png --task "Defend the center: shoot the enemies closing in from all directions."
"""

import argparse
import json

from .client import DOOM_TASK, VLAClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--checkpoint", required=True, help="the served checkpoint's directory (its tokenizer.json)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", required=True, help="the task; with --game doom, put into DOOM_TASK")
    ap.add_argument("--game", default="doom")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.game == "minecraft":
        client = VLAClient(a.base_url, a.checkpoint, game="minecraft", temperature=a.temperature,
                           resize_width=None, normalize_task=False)
    else:
        client = VLAClient(a.base_url, a.checkpoint, game=a.game, temperature=a.temperature)
    instruction = DOOM_TASK.format(task=a.task) if a.game == "doom" else a.task
    chunk = client.act(a.image, instruction, seed=a.seed)
    print(json.dumps({"dx": chunk.dx, "dy": chunk.dy, "dz": chunk.dz, "turn_deg": chunk.turn_deg,
                      "keys": chunk.keys, "flags": chunk.flags, "error": chunk.error, "text": chunk.text}, indent=1))


if __name__ == "__main__":
    main()
