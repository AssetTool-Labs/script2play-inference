import numpy as np

from script2play import DOOM_TASK, as_instruction, encode_frame, parse
from script2play.prompt import instruction_system


def test_doom_prompt():
    task = as_instruction(DOOM_TASK.format(task="defend the center"))
    assert instruction_system("doom", task, brief=True) == (
        "You are an experienced Roblox doom PC player, proficient in keyboard and mouse operations. "
        "Based on the current game screen, plan the next 200ms of actions, consisting of 4 steps. "
        "Each step is spaced 50ms apart.\n"
        "Your current task is: Control the Doom player to complete the task: defend the center.")


def test_minecraft_prompt():
    from script2play import VLAClient

    client = object.__new__(VLAClient)
    client.game, client.brief, client.normalize_task = "minecraft", True, False
    assert client.system_prompt("mine iron ore") == (
        "You are an experienced Minecraft PC player, proficient in keyboard and mouse operations. "
        "Based on the current game screen, plan the next 200ms of actions, consisting of 4 steps. "
        "Each step is spaced 50ms apart.\n"
        "Your current task is: mine iron ore")


def test_parse_chunk():
    c = parse("-150 0 0 ; LB ; LB W ; ; Left")
    assert (c.dx, c.dy, c.dz) == (-150, 0, 0)
    assert c.keys == [["LB"], ["LB", "W"], [], ["Left"]]
    assert not c.malformed


def test_parse_pads_and_flags():
    c = parse("12 3 0 ; D")
    assert c.keys == [["D"], [], [], []]
    assert parse("I would turn left").malformed
    assert parse("").keys == [[], [], [], []]


def test_steps():
    steps = parse("-150 7 0 ; LB ; LB W ; ; Left").steps()
    assert [s.t_ms for s in steps] == [0, 50, 100, 150]
    assert [s.dx for s in steps] == [-38, -38, -37, -37]
    assert [s.dy for s in steps] == [2, 2, 2, 1]
    assert [(s.pressed, s.released) for s in steps] == [
        (["LB"], []), (["W"], []), ([], ["LB", "W"]), (["Left"], [])]
    assert parse("0 0 0 ; ; ; ;").steps(held_before=["W"])[0].released == ["W"]
    assert parse("-150 0 0 ; ; ; ;").turn_deg == (-22.5, 0.0)


def test_act_gives_idle_chunk_when_unreachable():
    import requests

    from script2play import VLAClient

    client = object.__new__(VLAClient)
    client.base_url, client.model, client.session, client.timeout = "http://127.0.0.1:9/v1", "m", requests.Session(), 0.5
    client.game, client.brief, client.temperature, client.top_p, client.top_k = "doom", True, 0.8, 0.99, 0
    client.max_tokens, client.stop_token_ids = 48, [1, 2]
    client.resize_width, client.normalize_task = 832, True
    chunk = client.act(np.zeros((480, 640, 3), np.uint8), "Defend the center.", retries=1)
    assert chunk.error and chunk.malformed and chunk.keys == [[], [], [], []] and chunk.dx == 0


def test_encode_frame_width():
    from PIL import Image
    import base64
    import io

    url = encode_frame(np.zeros((480, 640, 3), np.uint8))
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.size == (832, 624)
