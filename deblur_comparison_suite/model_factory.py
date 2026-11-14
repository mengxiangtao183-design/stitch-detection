from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn


SUITE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SUITE_ROOT.parent
THIRD_PARTY = SUITE_ROOT / "third_party"


def _prepend(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _bind_third_party_basicsr(repo: Path) -> None:
    """Expose one vendored BasicSR tree without triggering its auto-import scans."""
    for name in tuple(sys.modules):
        if name == "basicsr" or name.startswith("basicsr."):
            del sys.modules[name]
    paths = {
        "basicsr": repo / "basicsr",
        "basicsr.models": repo / "basicsr" / "models",
        "basicsr.models.archs": repo / "basicsr" / "models" / "archs",
    }
    for name, directory in paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package


def _flatten_tensors(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        if value.ndim == 4:
            return [value]
        if value.ndim == 5:
            return [value[:, index] for index in range(value.shape[1])]
        return []
    if isinstance(value, dict):
        for key in ("img", "output", "restored", "result", "pred"):
            if key in value:
                tensors = _flatten_tensors(value[key])
                if tensors:
                    return tensors
        tensors: list[torch.Tensor] = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


class UnifiedDeblurModel(nn.Module):
    """Normalize third-party forward signatures without rewriting their networks."""

    def __init__(self, model: nn.Module, name: str, final_policy: str = "last") -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.final_policy = final_policy

    def forward_all(self, image: torch.Tensor) -> list[torch.Tensor]:
        raw = self.model(image)
        predictions = _flatten_tensors(raw)
        # Published multi-scale methods may additionally return 1/2x, 1/4x
        # predictions or kernel maps.  The common benchmark supervises only
        # full-resolution restoration images, rather than resizing a target
        # and unintentionally changing the loss definition per architecture.
        full_resolution = [
            tensor for tensor in predictions if tensor.shape[-2:] == image.shape[-2:]
        ]
        if full_resolution:
            predictions = full_resolution
        if not predictions:
            raise TypeError(f"{self.name} returned no NCHW image tensor: {type(raw)!r}")
        if self.final_policy == "mean_heads":
            return [torch.stack(predictions, dim=0).mean(dim=0)]
        return predictions

    def select_final(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        if self.final_policy == "first":
            return predictions[0]
        return predictions[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        predictions = self.forward_all(image)
        return self.select_final(predictions)


def _build_mbdnet_p_plus() -> nn.Module:
    _prepend(PROJECT_ROOT / "NAFNet")
    module = importlib.import_module("basicsr.models.archs.MBDNetP_arch")
    return module.MBDNetPPlus(img_channel=3, width=64)


def _build_nafnet() -> nn.Module:
    _prepend(PROJECT_ROOT / "NAFNet")
    module = importlib.import_module("basicsr.models.archs.NAFNet_arch")
    return module.NAFNet(
        img_channel=3,
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )


def _build_mprnet() -> nn.Module:
    _prepend(PROJECT_ROOT / "Deblurring")
    return importlib.import_module("MPRNet").MPRNet(
        in_c=3,
        out_c=3,
        n_feat=96,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
    )


def _build_restormer() -> nn.Module:
    _bind_third_party_basicsr(THIRD_PARTY / "Restormer")
    module = importlib.import_module("basicsr.models.archs.restormer_arch")
    return module.Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )


def _build_stripformer() -> nn.Module:
    _prepend(THIRD_PARTY / "Stripformer")
    return importlib.import_module("models.Stripformer").Stripformer()


def _build_banet() -> nn.Module:
    _prepend(THIRD_PARTY / "BANet")
    return importlib.import_module("models.BANet_model").BANet_model()


def _build_loformer() -> nn.Module:
    _bind_third_party_basicsr(THIRD_PARTY / "LoFormer")
    module = importlib.import_module("basicsr.models.archs.LoFormer_arch")
    return module.LoFormer(
        inp_channels=3,
        out_channels=3,
        dim=32,
        enc_blk_nums=[1, 2, 3],
        middle_blk_num=7,
        dec_blk_nums=[3, 2, 2],
        heads_enc=[1, 2, 4],
        heads_mid=8,
        window_size_enc=[8, 8, 8],
        window_size_mid=8,
        ffn_expansion_factor=2.66,
        bias=True,
        LayerNorm_type="WithBias",
        train_size=None,
    )


def _build_mimo_unet_plus() -> nn.Module:
    _prepend(THIRD_PARTY / "MIMO-UNet")
    module = importlib.import_module("models.MIMOUNet")
    return module.MIMOUNetPlus(num_res=20)


def _build_hinet() -> nn.Module:
    _bind_third_party_basicsr(THIRD_PARTY / "HINet_official")
    module = importlib.import_module("basicsr.models.archs.hinet_arch")
    return module.HINet(in_chn=3, wf=64, depth=5)


def _build_ufpnet() -> nn.Module:
    _bind_third_party_basicsr(THIRD_PARTY / "UFPDeblur")
    module = importlib.import_module("basicsr.models.archs.UFPNet_code_uncertainty_arch")
    return module.UFPNet_code_uncertainty(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 8],
        dec_blk_nums=[1, 1, 1, 1],
        kernel_size=19,
    )


def _build_sfnet() -> nn.Module:
    _prepend(THIRD_PARTY / "SFNet_official" / "Motion_deblurring")
    module = importlib.import_module("models.SFNet")
    return module.SFNet(mode=["train", "GOPRO"], num_res=16)


BUILDERS = {
    "mbdnet_p_plus": (_build_mbdnet_p_plus, "last"),
    "mprnet": (_build_mprnet, "first"),
    "nafnet": (_build_nafnet, "last"),
    "restormer": (_build_restormer, "last"),
    "stripformer": (_build_stripformer, "last"),
    "banet": (_build_banet, "last"),
    "loformer": (_build_loformer, "last"),
    "mimo_unet_plus": (_build_mimo_unet_plus, "last"),
    "hinet": (_build_hinet, "last"),
    "ufpnet": (_build_ufpnet, "last"),
    "sfnet": (_build_sfnet, "last"),
}


def available_models() -> tuple[str, ...]:
    return tuple(BUILDERS)


def build_model(name: str) -> UnifiedDeblurModel:
    key = name.lower().replace("-", "_")
    if key not in BUILDERS:
        raise KeyError(f"Unknown model {name!r}; choose from {', '.join(BUILDERS)}")
    builder, final_policy = BUILDERS[key]
    return UnifiedDeblurModel(builder(), key, final_policy)
