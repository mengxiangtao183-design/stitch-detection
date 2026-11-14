"""Single model factory used by training, evaluation and prediction."""

from __future__ import annotations

from typing import Dict


_ALIASES = {
    "unet": "unet",
    "u-net": "unet",
    "deeplab": "deeplabv3plus",
    "deeplabv3+": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "segformer": "segformer_b0",
    "segformer-b0": "segformer_b0",
    "segformer_b0": "segformer_b0",
    "gated-scnn": "gated_scnn",
    "gated_scnn": "gated_scnn",
    "pointrend": "pointrend_r50",
    "pointrend_r50": "pointrend_r50",
    "pidnet": "pidnet_s",
    "pidnet-s": "pidnet_s",
    "pidnet_s": "pidnet_s",
    "pspnet": "pspnet_r50",
    "pspnet-r50": "pspnet_r50",
    "pspnet_r50": "pspnet_r50",
    "opnet": "opnet",
    "fspnet": "fspnet",
    "fselnet": "fselnet",
}


def _architecture(config: Dict) -> str:
    value = str(config.get("architecture", "c3net")).strip().lower()
    return _ALIASES.get(value, value)


def is_comparison_model(config: Dict) -> bool:
    return _architecture(config) != "c3net"


def list_models():
    return tuple(sorted(set(_ALIASES.values())))


def build_model(config: Dict):
    architecture = _architecture(config)
    if architecture == "c3net":
        from models.c3net import C3Net
        return C3Net(config)

    from .standard_models import DeepLabV3Plus, GatedSCNN, PointRend, PSPNet, SegFormerB0, UNet
    builders = {
        "unet": UNet,
        "deeplabv3plus": DeepLabV3Plus,
        "segformer_b0": SegFormerB0,
        "gated_scnn": GatedSCNN,
        "pointrend_r50": PointRend,
        "pspnet_r50": PSPNet,
    }
    if architecture in builders:
        return builders[architecture](config)
    if architecture == "opnet":
        from .OPNet_model import OPNet_Wrapper
        return OPNet_Wrapper(config)
    if architecture == "fspnet":
        from .FSPNet_model import Model
        return Model(config)
    if architecture == "fselnet":
        from .fselnet_model import fselnet_Wrapper
        return fselnet_Wrapper(config)
    if architecture == "pidnet_s":
        from .pidnet_model import PIDNetSWrapper
        return PIDNetSWrapper(config)
    raise ValueError(
        f"Unknown model architecture '{architecture}'. Available: {', '.join(list_models())}"
    )
