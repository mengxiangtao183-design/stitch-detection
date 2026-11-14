from __future__ import annotations

import argparse
import json

import torch

from model_factory import available_models, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="One-process forward smoke test for a registered model")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke test")
    torch.manual_seed(42)
    model = build_model(args.model).cuda().eval()
    image = torch.rand(1, 3, args.size, args.size, device="cuda")
    with torch.no_grad():
        output = model(image)
    if output.shape != image.shape:
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}; expected {tuple(image.shape)}")
    print(json.dumps({
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "output_shape": list(output.shape),
    }))


if __name__ == "__main__":
    main()
