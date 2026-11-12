"""The system prompts the Script2Play checkpoints were trained on; a change here is a change to the
model's input.

``instruction_system(game, instruction, brief=True)`` is what the benchmarks use: "doom" for
Doom-like games (the default controls) and "minecraft" for Minecraft.
"""

GAME_CONTROLS = {
    'minecraft': (True, 'A survival sandbox. LB attacks and mines (hold it to break a block), '
                        'RB uses or places the held item, E opens and closes the inventory, Q '
                        'drops the held item, one-nine select a hotbar slot, W and S move '
                        'forward and back, A and D go left and right, Space jumps, Shift sneaks '
                        'and Ctrl sprints. With the inventory or a crafting screen open the '
                        'mouse moves the cursor, and LB and RB click.'),
}
# The name the prompt calls a game by, in place of "Roblox <name>".
GAME_TITLES = {'minecraft': 'Minecraft'}
DEFAULT_CONTROLS = (True, 'W and S move forward and back, A and D go left and right, Space '
                          'jumps, Shift sprints, E interacts, LB is the primary action and '
                          'RB the secondary one.')

_MOUSE_AIM = ('1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means '
              'move right, Y>0 means move down) and scroll amount Z (Z>0 means scroll up).')
_MOUSE_IDLE = ('1. **Mouse**: The camera does not follow the mouse in this game, so the leading '
               'X Y Z field is always 0 0 0.')

INSTRUCTION_SYSTEM = (
    'You are an experienced {game} PC player, proficient in {skills}. '
    'Based on the current game screen, plan the next 200ms of actions, consisting of 4 '
    'steps. Each step is spaced 50ms apart. Every step lasts 50ms from its start time until '
    'the next step begins.\n'
    '**Output Format**\n'
    '<|action_start|>X Y Z ; k1 k2 ; k3 ; k4 ; k5<|action_end|>\n'
    '**Explanation**\n'
    '{mouse}\n'
    '2. **Key Sequence**: Then list 4 groups of keys; within each group, keys are separated '
    "by spaces, and groups are separated by semicolons.\n"
    '   - Each group can contain up to 4 keys.\n'
    "   - If a group has no keys, leave it empty but keep the ';'.\n"
    '3. Only output a plain string that conforms to the above format \u2014 no line breaks and no '
    'quotation marks.\n'
    '**Key Naming Rules**\n'
    "- Number keys '1\u20139': use lowercase English words, e.g., 'one' represents the '1' key "
    'on the keyboard.\n'
    "- Function keys 'F1\u2013F12': use capitalized English words, e.g., 'One' represents 'F1', "
    "'Two' represents 'F2', and so on.\n"
    '- Other keys (letters, Shift, Tab, Space, etc.): use the real keyboard name with an '
    "initial capital letter, e.g., 'A', 'D', 'Shift', 'Space', etc.\n"
    "- Mouse buttons: 'LB', 'RB' and 'MB' for the left, right and middle button; arrow keys "
    "are 'Up', 'Down', 'Left' and 'Right'.\n"
    '**Controls**\n'
    '{controls}\n'
    'Your current task is: {instruction}'
)

INSTRUCTION_SYSTEM_BRIEF = (
    'You are an experienced {game} PC player, proficient in {skills}. '
    'Based on the current game screen, plan the next 200ms of actions, consisting of 4 '
    'steps. Each step is spaced 50ms apart.\n'
    'Your current task is: {instruction}'
)

# The brief prompt with a game's own control description.
INSTRUCTION_SYSTEM_BRIEF_CONTROLS = (
    'You are an experienced {game} PC player, proficient in {skills}. '
    'Based on the current game screen, plan the next 200ms of actions, consisting of 4 '
    'steps. Each step is spaced 50ms apart.\n'
    'Controls: {controls}\n'
    'Your current task is: {instruction}'
)


def instruction_system(game: str, instruction: str, *, brief: bool = False, controls: str | None = None,
                       aims: bool | None = None) -> str:
    """The instruction prompt for a game label.

    ``controls`` (a game's own control description) replaces the label's; with ``brief`` it also
    adds the Controls line (INSTRUCTION_SYSTEM_BRIEF_CONTROLS). ``aims`` says whether the mouse
    moves the view or a cursor, when the label does not. A label other than those above is
    called "Roblox <label>" with the default controls, as the training data named its games."""
    known_aims, known_controls = GAME_CONTROLS.get(game, DEFAULT_CONTROLS)
    aims = known_aims if aims is None else aims
    template = INSTRUCTION_SYSTEM_BRIEF if brief else INSTRUCTION_SYSTEM
    if controls is not None and brief:
        template = INSTRUCTION_SYSTEM_BRIEF_CONTROLS
    controls = known_controls if controls is None else controls
    title = GAME_TITLES.get(game)
    game = game or 'the game'
    if title:
        game = title
    elif game != 'the game' and not game.lower().startswith('roblox'):
        game = f'Roblox {game}'
    return template.format(
        game=game,
        skills='keyboard and mouse operations' if aims else 'keyboard controls',
        mouse=_MOUSE_AIM if aims else _MOUSE_IDLE,
        controls=controls,
        instruction=instruction or '')
