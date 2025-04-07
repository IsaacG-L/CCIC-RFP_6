# Native imports
from typing import Tuple


class BuildFroggerConfig:
    num_envs : int = 4
    env_name : str = "ALE/Frogger-v5"
    render_mode : str = "rgb_array"
    max_episode_steps : int = None
    disable_env_checker : bool = None
    full_action_space : bool = False

class PreprocessingFroggerConfig:
    noop_max : int = 30
    frame_skip : int = 1
    screen_size: int | Tuple[int, int] = 84
    terminal_on_life_loss : bool = False
    grayscale_obs : bool = True
    grayscale_newaxis : bool = True
    scale_obs : bool = False
