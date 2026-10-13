"""Shared frozen BERT language encoder and config fields for the language-conditioned policies."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@dataclass
class LanguageConditioningConfig:
    """Config fields mixed into LangACTConfig / LangDiffusionConfig.

    Args:
        language_model_name: Hugging Face model id of the (frozen) text encoder and its tokenizer.
        tokenizer_max_length: Task strings are padded / truncated to this many tokens.
        language_pooling: How to turn BERT's token outputs into one task embedding.
            "mean": attention-masked mean over tokens, "cls": the [CLS] token output.
    """

    language_model_name: str = "google-bert/bert-base-uncased"
    tokenizer_max_length: int = 48
    language_pooling: str = "mean"

    def validate_language_config(self) -> None:
        if self.language_pooling not in ("mean", "cls"):
            raise ValueError(f"`language_pooling` must be 'mean' or 'cls', got '{self.language_pooling}'.")


class LanguageEncoder(nn.Module):
    """Encodes tokenized task strings into a single (B, hidden_size) embedding with a frozen BERT.

    BERT is deliberately kept outside the module tree: it has no parameters in the policy, so it is never
    optimized, never saved in policy checkpoints, and always (re)loaded from the Hugging Face cache by name.
    It follows the device of its inputs.
    """

    def __init__(self, config: LanguageConditioningConfig):
        super().__init__()
        self.pooling = config.language_pooling
        bert = AutoModel.from_pretrained(config.language_model_name, add_pooling_layer=False)
        bert.requires_grad_(False)
        bert.eval()
        self.__dict__["bert"] = bert  # bypass nn.Module registration (see class docstring)
        self.output_dim = bert.config.hidden_size

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.bert.device != input_ids.device:
            self.bert.to(input_ids.device)
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_batch(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_LANGUAGE_TOKENS not in batch:
            raise KeyError(
                f"'{OBS_LANGUAGE_TOKENS}' is missing from the batch. Language-conditioned policies need the "
                "task string tokenized by their preprocessor (make sure the batch has a 'task' entry)."
            )
        return self.forward(batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
