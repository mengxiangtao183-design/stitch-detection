import torch
import os
from collections import OrderedDict


def load_checkpoint_file(weights, load_state_dict=True):
    checkpoint = torch.load(weights, map_location='cpu')
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get('checkpoint_format') == 'safetensors_manifest_v1'
    ):
        if not load_state_dict:
            return checkpoint
        from safetensors.torch import load_file

        state_path = os.path.join(
            os.path.dirname(weights), checkpoint['state_dict_file']
        )
        checkpoint = dict(checkpoint)
        checkpoint['state_dict'] = load_file(state_path, device='cpu')
    return checkpoint

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights, strict=True):
    checkpoint = load_checkpoint_file(weights)
    state_dict = checkpoint["state_dict"]
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        if not any(key.startswith("module.") for key in state_dict):
            raise error
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        incompatible = model.load_state_dict(new_state_dict, strict=strict)
    return incompatible


def load_checkpoint_partial(model, weights):
    checkpoint = load_checkpoint_file(weights)
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    filtered_state = OrderedDict()
    skipped = []
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
            filtered_state[normalized_key] = value
        else:
            skipped.append(normalized_key)
    incompatible = model.load_state_dict(filtered_state, strict=False)
    return incompatible, skipped


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = load_checkpoint_file(weights, load_state_dict=False)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = load_checkpoint_file(weights, load_state_dict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
