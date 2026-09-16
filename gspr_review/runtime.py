"""Shared data/runtime contracts inherited from the verified communication experiment."""
from gspr_communication.runtime import (ROOT, sha256, verify_frozen, new_output, seed_all,
    device, make_loader, model_input, write_json)
from gspr_communication.runtime import load_config as _load_config


def load_config(experiment, frontend_config):
    import copy
    from .resources import resolve_weather
    options, hypes = _load_config(experiment, frontend_config)
    if 'weather_augmentation' in hypes:
        weather = resolve_weather(hypes['weather_augmentation'], ROOT)
        # Persist the resolved path with the run, for the identical evaluation protocol.
        options['weather_augmentation'] = copy.deepcopy(weather)
        hypes['weather_augmentation'] = weather
        if weather.get('mode') in ('physics_fog', 'mixed_physics_weather'):
            print('Fog lookup directory:', weather['physics_fog']['lookup_dir'], flush=True)
    return options, hypes


def load_model(hypes, options, checkpoint, target):
    import torch
    from .model import ReviewModel
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model = ReviewModel(hypes['model']['args'], options['communication'], options['review'])
    model.load_frontend(state)
    if sha256(checkpoint) != digest:
        raise RuntimeError('Frontend checkpoint changed during load')
    return model.to(target), digest


def load_reviewer(model, checkpoint, options, frontend_hash, frontend_config):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state['frontend_sha256'] != frontend_hash or
            state['frontend_config_sha256'] != sha256(frontend_config) or
            state['communication'] != options['communication'] or state['review'] != options['review']):
        raise ValueError('Reviewer/frontend/protocol mismatch; use saved run config')
    model.reviewer.load_state_dict(state['reviewer'], strict=True)


def input_branch(ego, options, branch=None):
    return model_input(ego, dict(options, lidar_key=branch or options['lidar_key']))
