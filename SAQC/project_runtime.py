"""Shared project loading and immutable frozen-F contract for SAQC."""
from __future__ import annotations

import json
from pathlib import Path

from gspr_communication.runtime import device, sha256, verify_frozen
from gspr_evidence import runtime as er
from local_fusion_detector_adaptation.candidate_audit import _load_arm
from local_fusion_v3 import runtime as v3rt
from .offline_weather import DEFAULT_ROOT


def load_frozen_f(args, *, hypes_override=None):
    """Load frozen frontend, v3 source and saved F arm with identity checks."""
    verify_frozen()
    options, base_hypes = er.load_config(
        args.v3_config, args.frontend_config)
    hypes = base_hypes if hypes_override is None else hypes_override
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)
    source, checkpoint = v3rt.load(
        args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError(
            'SAQC benchmark expects the residual v3 checkpoint used by F')
    run = Path(args.run).resolve()
    arm = _load_arm(
        run / 'F.pth', source, model.engine.base, target, False)
    arm.requires_grad_(False).eval()
    protocol_path = run / 'protocol.json'
    if protocol_path.is_file():
        protocol = json.loads(
            protocol_path.read_text(encoding='utf-8'))
        expected = protocol.get('frontend_sha256')
        if expected is not None and expected != frontend_digest:
            raise ValueError(
                'frontend checkpoint differs from saved F run')
        expected_v3 = protocol.get('v3_checkpoint_sha256')
        if (expected_v3 is not None
                and expected_v3 != sha256(args.v3_checkpoint)):
            raise ValueError(
                'v3 checkpoint differs from saved F run')
    else:
        protocol = {}
    return dict(
        options=options,
        hypes=hypes,
        target=target,
        model=model,
        arm=arm,
        frontend_sha256=frontend_digest,
        v3_checkpoint_sha256=sha256(args.v3_checkpoint),
        f_checkpoint_sha256=sha256(run / 'F.pth'),
        protocol=protocol,
    )


def add_common_arguments(parser):
    parser.add_argument(
        '--run', required=True,
        help='completed local_fusion_detector_adaptation run containing F.pth')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--hidden-channels', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--weather-dataset-root', default=DEFAULT_ROOT,
                        help='Complete fixed epoch-0 physics dataset under /data')
    parser.add_argument(
        '--smoke', type=int, default=0,
        help='limit frames per condition; 0 means full split')


def validate_common(args):
    if args.patch_size < 1 or args.patch_size % 2 != 1:
        raise ValueError(
            '--patch-size must be a positive odd integer')
    if args.hidden_channels < 1:
        raise ValueError('--hidden-channels must be positive')
    if args.beta < 0:
        raise ValueError('--beta must be nonnegative')
    if args.smoke < 0:
        raise ValueError('--smoke must be nonnegative')
