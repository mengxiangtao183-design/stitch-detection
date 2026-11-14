"""Export a transition checkpoint into the deployable pure-TEPR architecture."""

import argparse
from pathlib import Path

import torch

from config import Config
from MPRNet import build_mprnet_from_config
from utils.model_utils import load_checkpoint_file, load_checkpoint_partial


def parse_args():
    parser = argparse.ArgumentParser(
        description='Strip the frozen legacy recovery branch from a TEPR transition checkpoint.'
    )
    parser.add_argument('--config', required=True, type=str,
                        help='Pure TEPR YAML: STAGE3_TEPR=true, STAGE3_TEPR_TRANSITION=false.')
    parser.add_argument('--weights', required=True, type=str,
                        help='Completed transition checkpoint with alpha equal to 1.')
    parser.add_argument('--output', required=True, type=str,
                        help='New output checkpoint for the pure TEPR model.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(args.config)
    model = build_mprnet_from_config(config.MODEL)
    if not model.stage3_tepr_is_pure:
        raise ValueError(
            'The export config must enable pure TEPR and must not enable '
            'STAGE3_TEPR_TRANSITION.'
        )

    checkpoint = load_checkpoint_file(args.weights)
    state_dict = checkpoint['state_dict']
    alpha = state_dict.get(
        'stage3_tepr_transition_alpha',
        state_dict.get('module.stage3_tepr_transition_alpha'),
    )
    if alpha is None or float(alpha.detach().cpu().item()) < 1.0 - 1e-8:
        raise ValueError(
            'Refusing to export: the transition checkpoint has not reached alpha=1. '
            'Train the transition model through its migration schedule first.'
        )

    incompatible, skipped = load_checkpoint_partial(model, args.weights)
    legacy_keys = [
        key for key in model.state_dict()
        if key.startswith('stage3_orsnet.') or key.startswith('tail.')
    ]
    if legacy_keys:
        raise RuntimeError('Pure TEPR export still contains legacy recovery parameters.')

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': int(checkpoint.get('epoch', 0)),
        'state_dict': model.state_dict(),
        'optimizer_lrs': [],
        'architecture': 'pure_tepr',
        'source_transition_checkpoint': str(args.weights),
        'transition_alpha': 1.0,
    }
    torch.save(payload, output)
    print(
        f'Exported pure TEPR checkpoint: {output}\n'
        f'missing={len(incompatible.missing_keys)}, '
        f'unexpected={len(incompatible.unexpected_keys)}, '
        f'skipped_shape={len(skipped)}'
    )


if __name__ == '__main__':
    main()
