from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion import DiffusionConfig

from ..language_encoder import LanguageConditioningConfig


@PreTrainedConfig.register_subclass("lang_diffusion")
@dataclass
class LangDiffusionConfig(DiffusionConfig, LanguageConditioningConfig):
    """Diffusion Policy conditioned on a BERT embedding of the task string.

    The pooled task embedding is projected to `language_embed_dim` and concatenated (once, not per observation
    step) to the U-Net's FiLM global conditioning vector:
    [obs features x n_obs_steps, language embedding].
    All Diffusion options (see `DiffusionConfig`) and language options (see `LanguageConditioningConfig`) apply.

    Args:
        language_embed_dim: Size of the projected language embedding fed to FiLM.
    """

    language_embed_dim: int = 128

    def __post_init__(self):
        super().__post_init__()
        self.validate_language_config()
