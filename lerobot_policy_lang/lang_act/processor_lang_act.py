from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from ..processing import make_language_conditioned_pre_post_processors
from .configuration_lang_act import LangACTConfig


def make_lang_act_pre_post_processors(
    config: LangACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_language_conditioned_pre_post_processors(config, dataset_stats)
