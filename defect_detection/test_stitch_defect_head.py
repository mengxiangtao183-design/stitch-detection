"""CPU smoke test for the post-C3Net six-unit defect detector."""

from pathlib import Path
import sys

import torch

# Running a script from this subdirectory changes the import root on
# Windows/Python; add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.stitch_defect_head import DEFECT_CLASSES, StitchDefectHead, multilabel_defect_loss


def main() -> None:
    torch.manual_seed(7)
    head = StitchDefectHead().train()
    final_mask = torch.randn(2, 1, 320, 320, requires_grad=True)
    centerline = torch.randn(2, 1, 320, 320)
    rgb = torch.randn(2, 3, 320, 320)
    output = head(final_mask, centerline, rgb)
    assert output["unit_logits"].shape == (2, 6, 4)
    assert output["image_logits"].shape == (2, 4)
    assert output["geometry"].shape == (2, 6, 7)
    target = torch.zeros_like(output["unit_logits"])
    target[0, 2, 0] = 1  # one annotated skipped stitch
    loss = multilabel_defect_loss(output["unit_logits"], target)
    loss.backward()
    assert final_mask.grad is not None
    print("PASS", {"classes": DEFECT_CLASSES, "loss": round(float(loss), 6)})


if __name__ == "__main__":
    main()
