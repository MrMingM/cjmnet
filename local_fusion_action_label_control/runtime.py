"""Reuse source loader/predict methods with content checks and read-only input.

The original constructor couples model loading to writable Manifest and Git
checks. This adapter uses its actual public dependencies for initialization;
it never patches the original class and never calls its constructor.
"""
from pathlib import Path
from .common import digest, read_json
from .inputs import Source
from local_fusion_action_utility_audit.common import Runtime as OriginalRuntime, assert_development_paths


class ReplayRuntime(OriginalRuntime):
    def __init__(self, source_run):
        from gspr_communication.runtime import device, verify_frozen
        from gspr_evidence import runtime as er
        from local_fusion_v3 import runtime as v3rt
        from local_fusion_task_source_oracle.oracle import _load_b0_arm, _verify_b0_source_contract
        self.source = Source(source_run)
        self.source.verify_dependencies()
        self.run = Path(source_run)
        self.protocol = self.source.protocol
        self.spec = self.source.spec
        verify_frozen()
        inputs = self.protocol['inputs']
        self.b0 = read_json(Path(inputs['b0_run']) / 'protocol.json')
        _verify_b0_source_contract(self.b0)
        self.options, self.hypes = er.load_config(inputs['v3_config'], inputs['frontend_config'])
        assert_development_paths(self.hypes['root_dir'], self.hypes['validate_dir'])
        self.target = device()
        self.model, frontend_hash = er.load_model(self.hypes, self.options, inputs['frontend_checkpoint'], self.target)
        self.model.requires_grad_(False).eval()
        contract = v3rt.contract(self.options, inputs['frontend_config'], frontend_hash)
        if self.b0['frontend_sha256'] != frontend_hash or self.b0['v3_contract'] != contract:
            raise ValueError('B0 frontend/source contract differs')
        v3_hash = digest(inputs['v3_checkpoint'])
        if self.b0['v3_checkpoint_sha256'] != v3_hash:
            raise ValueError('B0 v3 checkpoint differs')
        source, state = v3rt.load(inputs['v3_checkpoint'], contract, self.target)
        if state['variant'] != 'residual':
            raise ValueError('Expected residual v3 checkpoint')
        self.shared_arm = _load_b0_arm(Path(inputs['b0_run']), 'Shared.pth', source, self.target, 'shared', v3_hash)
        self.lidar_range = self.hypes['postprocess']['anchor_args']['cav_lidar_range']
        if any(p.requires_grad for module in (self.model, self.shared_arm) for p in module.parameters()):
            raise RuntimeError('Frozen model contains trainable parameters')
