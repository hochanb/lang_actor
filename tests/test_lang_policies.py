"""Shape / conditioning sanity checks for lang_act and lang_diffusion on synthetic PushT-like features."""

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

import lerobot_policy_lang  # noqa: F401  (registers the policy types)

IMAGE_KEY = "observation.image"
STATE_DIM, ACTION_DIM, B = 2, 2, 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _features():
    inputs = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return inputs, outputs


def _stats():
    def s(d):
        return {
            "mean": torch.zeros(d),
            "std": torch.ones(d),
            "min": -torch.ones(d),
            "max": torch.ones(d),
        }

    img = {k: v.view(1, 1, 1).expand(3, 1, 1) for k, v in s(1).items()}
    return {IMAGE_KEY: img, OBS_STATE: s(STATE_DIM), ACTION: s(ACTION_DIM)}


def _make(policy_type, **overrides):
    inputs, outputs = _features()
    cfg_cls = lerobot_policy_lang.LangACTConfig if policy_type == "lang_act" else lerobot_policy_lang.LangDiffusionConfig
    cfg = cfg_cls(input_features=inputs, output_features=outputs, device=DEVICE, **overrides)
    if policy_type == "lang_diffusion":
        cfg.crop_shape = (84, 84)
        cfg.num_inference_steps = 5
    policy = get_policy_class(policy_type)(cfg).to(DEVICE)
    pre, post = make_pre_post_processors(cfg, dataset_stats=_stats())
    return cfg, policy, pre, post


def _batch(cfg, tasks, n_obs=None):
    obs_shape = (B,) if n_obs is None else (B, n_obs)
    horizon = cfg.chunk_size if hasattr(cfg, "chunk_size") else cfg.horizon
    return {
        IMAGE_KEY: torch.rand(*obs_shape, 3, 96, 96),
        OBS_STATE: torch.randn(*obs_shape, STATE_DIM),
        ACTION: torch.randn(B, horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(B, horizon, dtype=torch.bool),
        "task": tasks,
    }


@pytest.mark.parametrize("policy_type", ["lang_act", "lang_diffusion"])
def test_train_step_and_language_sensitivity(policy_type):
    cfg, policy, pre, _ = _make(policy_type)
    n_obs = cfg.n_obs_steps if policy_type == "lang_diffusion" else None
    tasks = ["push the t block", "pick up the red cube", "open the drawer"]

    # Training step: loss is finite, BERT stays frozen, the language projection gets gradients.
    policy.train()
    batch = pre(_batch(cfg, tasks, n_obs))
    loss, _ = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss)
    enc = policy.model.language_encoder if policy_type == "lang_act" else policy.diffusion.language_encoder
    assert not enc.bert.training, "frozen BERT must stay in eval mode"
    assert all(p.grad is None for p in enc.bert.parameters())
    assert not any("bert" in n for n, _ in policy.named_parameters()), "BERT must not be a policy parameter"
    proj = policy.model.encoder_language_input_proj if policy_type == "lang_act" else policy.diffusion.language_proj
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0

    # Inference: same observations + same noise, different task string -> different actions.
    policy.eval()
    obs = _batch(cfg, tasks, n_obs)
    obs.pop(ACTION)
    obs.pop("action_is_pad")
    kwargs = {}
    if policy_type == "lang_diffusion":
        kwargs["noise"] = torch.randn(B, cfg.horizon, ACTION_DIM, device=DEVICE)
    a1 = policy.predict_action_chunk(pre(dict(obs)), **kwargs)
    obs["task"] = ["completely different instruction"] * B
    a2 = policy.predict_action_chunk(pre(dict(obs)), **kwargs)
    assert a1.shape[0] == B and a1.shape[-1] == ACTION_DIM
    assert not torch.allclose(a1, a2), "actions should depend on the language instruction"


@pytest.mark.parametrize("policy_type", ["lang_act", "lang_diffusion"])
def test_select_action_rollout(policy_type):
    cfg, policy, pre, post = _make(policy_type)
    policy.reset()
    for _ in range(3):
        obs = {
            IMAGE_KEY: torch.rand(B, 3, 96, 96),
            OBS_STATE: torch.randn(B, STATE_DIM),
            "task": ["push the t block"] * B,
        }
        action = post(policy.select_action(pre(obs)))
        assert action.shape == (B, ACTION_DIM)


@pytest.mark.parametrize("policy_type", ["lang_act", "lang_diffusion"])
def test_save_load_roundtrip(policy_type, tmp_path):
    cfg, policy, pre, post = _make(policy_type)
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    from safetensors import safe_open

    with safe_open(tmp_path / "model.safetensors", "pt") as f:
        saved_keys = list(f.keys())
    assert not any("bert" in k or "language_encoder" in k for k in saved_keys), "BERT must not be checkpointed"
    loaded = get_policy_class(policy_type).from_pretrained(tmp_path)
    loaded_pre, _ = make_pre_post_processors(loaded.config, pretrained_path=tmp_path)
    for (n, p), (_, q) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(p.cpu(), q.cpu()), n
    batch = loaded_pre({IMAGE_KEY: torch.rand(3, 96, 96), OBS_STATE: torch.randn(STATE_DIM), "task": "push"})
    assert "observation.language.tokens" in batch
