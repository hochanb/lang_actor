"""Language-conditioned ACT / Diffusion Policy plugins for LeRobot.

Importing this package registers the `lang_act` and `lang_diffusion` policy types.
"""

from .lang_act import LangACTConfig
from .lang_diffusion import LangDiffusionConfig

__all__ = ["LangACTConfig", "LangDiffusionConfig"]
