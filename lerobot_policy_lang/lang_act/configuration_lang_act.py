from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act import ACTConfig

from ..language_encoder import LanguageConditioningConfig


@PreTrainedConfig.register_subclass("lang_act")
@dataclass
class LangACTConfig(ACTConfig, LanguageConditioningConfig):
    """ACT conditioned on a BERT embedding of the task string.

    The pooled task embedding is projected to `dim_model` and added as one extra 1D token to the transformer
    encoder input: [latent, language, (robot_state), (env_state), (image_feature_map_pixels)].
    All ACT options (see `ACTConfig`) and language options (see `LanguageConditioningConfig`) apply.
    """

    def __post_init__(self):
        super().__post_init__()
        self.validate_language_config()
