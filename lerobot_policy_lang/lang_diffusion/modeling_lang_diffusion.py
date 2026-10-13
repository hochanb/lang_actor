"""Diffusion Policy conditioned on a BERT task embedding.

Identical to `lerobot.policies.diffusion.DiffusionPolicy` except that the U-Net's FiLM global conditioning is
[obs features x n_obs_steps, projected language embedding].
"""

from collections import deque

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from ..language_encoder import LanguageEncoder
from .configuration_lang_diffusion import LangDiffusionConfig

LANGUAGE_KEYS = (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)


class LangDiffusionPolicy(DiffusionPolicy):
    config_class = LangDiffusionConfig
    name = "lang_diffusion"

    def __init__(self, config: LangDiffusionConfig, **kwargs):
        # Same as DiffusionPolicy.__init__ but builds LangDiffusionModel.
        require_package("diffusers", extra="diffusion")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self._queues: dict[str, deque[Tensor]] = {}

        self.diffusion = LangDiffusionModel(config)

        self.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Same as DiffusionPolicy.predict_action_chunk, but the language tokens (which are not part of the
        observation history queues) are carried over from the current batch."""
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            stacked = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            stacked.update({k: batch[k] for k in LANGUAGE_KEYS if k in batch})
            batch = stacked
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.diffusion.generate_actions(batch, noise=noise)


class LangDiffusionModel(DiffusionModel):
    def __init__(self, config: LangDiffusionConfig) -> None:
        super().__init__(config)
        self.language_encoder = LanguageEncoder(config)
        self.language_proj = nn.Linear(self.language_encoder.output_dim, config.language_embed_dim)

        # Rebuild the U-Net with room for the language embedding in the FiLM conditioning.
        global_cond_dim = self._obs_cond_dim() * config.n_obs_steps + config.language_embed_dim
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    def _obs_cond_dim(self) -> int:
        """Per-observation-step conditioning size, mirroring DiffusionModel.__init__."""
        dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            encoders = (
                self.rgb_encoder
                if self.config.use_separate_rgb_encoder_per_camera
                else [self.rgb_encoder] * len(self.config.image_features)
            )
            dim += sum(encoder.feature_dim for encoder in encoders)
        if self.config.env_state_feature:
            dim += self.config.env_state_feature.shape[0]
        return dim

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        obs_cond = super()._prepare_global_conditioning(batch)  # (B, obs_cond_dim * n_obs_steps)
        language_cond = self.language_proj(self.language_encoder.encode_batch(batch))  # (B, language_embed_dim)
        return torch.cat([obs_cond, language_cond], dim=-1)
