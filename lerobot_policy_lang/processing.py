"""Pre/post-processor pipeline shared by the language-conditioned policies."""

from typing import Any

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .language_encoder import LanguageConditioningConfig


def make_language_conditioned_pre_post_processors(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """The default ACT/DP pipeline plus a step that tokenizes `batch["task"]` with BERT's tokenizer.

    `config` is a policy config that also inherits `LanguageConditioningConfig`.
    The tokenizer step writes `observation.language.tokens` / `observation.language.attention_mask`.
    """
    if not isinstance(config, LanguageConditioningConfig):
        raise TypeError(f"{type(config).__name__} must inherit LanguageConditioningConfig.")
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        TokenizerProcessorStep(
            tokenizer_name=config.language_model_name,
            max_length=config.tokenizer_max_length,
            padding="max_length",
            padding_side="right",
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
