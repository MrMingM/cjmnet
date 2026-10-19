"""Instrument the unchanged documented official train.main; do not replace its loop."""
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

from auditlib import (DEFAULTS, HERE, SELECTION, atomic_json, bound_json, complete,
    frozen_c, identity, jsonable, patched, read_json, request, runtime, sha256, verify_train)


def atomic_torch(path, value, save):
    temporary = path.with_name(path.name + '.tmp')
    save(value, temporary)
    # Flush serialized state before publishing the name; completion journals are published last.
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_config(args, output):
    import json
    import yaml
    from opencood.hypes_yaml import yaml_utils
    from run import config_difference
    original = Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    text = original.read_text(encoding='utf-8')
    old = yaml.safe_load(text)
    changes = dict(root_dir=str(Path(args.train_root) / 'train'),
                   validate_dir=str(Path(args.view_root) / 'validate'))
    for key, value in changes.items():
        text, count = re.subn('^' + key + r':[^\n]*$', lambda m: key + ': ' + json.dumps(value), text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected one official config field: ' + key)
    if set(config_difference(old, yaml.safe_load(text))) != set(changes):
        raise ValueError('Unexpected training config differences')
    destination = output / 'config.yaml'
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Training config changed')
    destination.write_text(text, encoding='utf-8', newline='\n')
    h = yaml_utils.load_yaml(str(destination))
    if (h['train_params'] != dict(batch_size=8, epoches=20, eval_freq=1, save_freq=1)
            or h['fusion']['core_method'] != 'IntermediateFusionDatasetV2XR'
            or not h['fusion']['args']['proj_first'] or not h['fov']
            or h['train_sim'] or h['eval_sim'] or 'max_cav' in h['train_params']):
        raise ValueError('Official Clean training flags changed')
    return destination, h


def run(args, own):
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    frozen = frozen_c(args)
    train_manifest = verify_train(args)
    verify_clean_loss_view(args, evidence)
    train_keys = {(s['scene'], t) for s in train_manifest['scenes'] for t in s['timestamps']}
    if train_keys & {tuple(k) for k in evidence['b']['selected_frame_keys']}:
        raise ValueError('Train and Clean validation physical frame keys overlap')
    runtime(args, evidence['a'])
    import numpy as np
    import torch
    from observe import module_inventory
    if not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('Require one visible HIP GPU, addressed internally as cuda:0')
    output = Path(args.output_root)
    request(output, dict(stage='D-1D', own_sources=own, train_manifest_identity=train_manifest['identity'],
        roots={k: getattr(args, k) for k in DEFAULTS}, smoke=args.smoke,
        policy=SELECTION, workers=16, physical_gpu=os.environ['ROCR_VISIBLE_DEVICES']), args.resume)
    bound_json(output / 'selection_policy.json', SELECTION)  # Before model construction or any gradient step.
    config, hypes = train_config(args, output)
    import platform
    contract = jsonable(dict(stage='D-1D', sources=own, policy=SELECTION,
        train_manifest_sha256=sha256(Path(args.train_root) / 'manifest.json'),
        train_manifest_identity=train_manifest['identity'], config_sha256=sha256(config),
        train_data_acceptance=train_manifest['train_data_acceptance'],
        original_yaml_sha256=evidence['a']['model']['original_yaml_sha256'],
        source_commit=evidence['a']['upstream_commit'], source_hashes=evidence['a']['upstream_source_hashes'],
        d1a=evidence['d1a_artifact_sha256'], d1b=evidence['d1b_artifact_sha256'], d1c=frozen['hashes'],
        frozen_d1c_gt_identity=frozen['gt']['identity'], parsed_config=hypes,
        max_cav=7, entrypoint='opencood/tools/train.py:main', initializer='official create_model, random seed 20261017',
        prohibited_initializers=['OPV2V', 'V2X-R author checkpoint'],
        loader=dict(batch_size=8, workers=16, shuffle_train=True, shuffle_val_loss=True,
                    drop_last_train=True, drop_last_val_loss=True, pin_memory=True, prefetch_factor=4,
                    val_collate='train_dataset.collate_batch_train', AP_frames=829,
                    train_frames=train_manifest['counts']['cooperative_frames'],
                    nominal_train_frames_per_epoch=8080, drop_last_train_frames=3),
        scheduler='unchanged official setup_lr_schedular; unchanged end-of-epoch scheduler.step(zero_based_epoch)',
        resume='last fully committed epoch; full model/optimizer/scheduler/Python/NumPy/Torch CPU+CUDA RNG restored',
        environment=dict(versions=evidence['a']['versions'], HIP=torch.version.hip, python=sys.version,
            platform=platform.platform(), gpu=torch.cuda.get_device_name(0), physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'],
            PYTHONHASHSEED=os.environ['PYTHONHASHSEED'],
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        smoke_only=args.smoke, no_test=True))
    contract['identity'] = identity(contract)
    bound_json(output / 'contract.json', contract)
    state_dir = output / 'resume_states'
    state_dir.mkdir(exist_ok=True)
    commits = []
    for epoch in range(1, 21):
        path = output / 'epochs' / f'{epoch:02d}' / 'complete.json'
        if not path.exists():
            break
        row = complete(path)
        if row['epoch'] != epoch or row['contract_identity'] != contract['identity']:
            raise ValueError('Epoch commit contract/order differs')
        if sha256(output / f'net_epoch{epoch}.pth') != row['checkpoint_sha256']:
            raise ValueError('Committed checkpoint changed')
        if sha256(row['clean_ap_summary']) != row['clean_ap_summary_sha256']:
            raise ValueError('Committed Clean AP result changed')
        if sha256(state_dir / f'{epoch:02d}.pth') != row['resume_state_sha256']:
            raise ValueError('Committed optimizer/scheduler/RNG state changed')
        commits.append(row)
    completed = len(commits)
    if any((output / 'epochs' / f'{i:02d}' / 'complete.json').exists() for i in range(completed + 2, 21)):
        raise ValueError('Non-contiguous committed epochs')
    if completed == 20:
        from evaluation import selection
        chosen = selection(args)
        finish_training(output, contract, chosen, args)
        print('All 20 epochs already committed. Selection verified; no training rerun.', flush=True)
        return
    saved_state = None
    if completed or (state_dir / '00.json').exists():
        state_path = state_dir / f'{completed:02d}.pth'
        if not completed:
            initial = read_json(state_dir / '00.json')
            if initial['contract_identity'] != contract['identity'] or initial['sha256'] != sha256(state_path):
                raise ValueError('Initial random state checkpoint changed')
        saved_state = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved_state['contract_identity'] != contract['identity'] or saved_state['completed_epoch'] != completed:
            raise ValueError('Resume state contract/epoch differs')
    random.seed(SELECTION['random_seed'])
    np.random.seed(SELECTION['random_seed'])
    torch.manual_seed(SELECTION['random_seed'])
    torch.cuda.manual_seed_all(SELECTION['random_seed'])
    entry = Path(args.runtime_root) / 'opencood/tools/train.py'
    spec = importlib.util.spec_from_file_location('_d1d_official_train', entry)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    utils = official.train_utils
    objects, observations = {}, {}
    original_save = torch.save
    original_create, original_loss = utils.create_model, utils.create_loss
    original_optimizer, original_scheduler = utils.setup_optimizer, utils.setup_lr_schedular
    original_loader, original_writer = official.DataLoader, official.SummaryWriter

    def rng():
        return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

    def restore_rng(row):
        random.setstate(row['python'])
        np.random.set_state(row['numpy'])
        torch.set_rng_state(row['cpu'])
        torch.cuda.set_rng_state_all(row['cuda'])

    def save_state(epoch):
        path = state_dir / f'{epoch:02d}.pth'
        state = dict(contract_identity=contract['identity'], completed_epoch=epoch,
            model=objects['model'].state_dict(), optimizer=objects['optimizer'].state_dict(),
            scheduler={k: v for k, v in objects['scheduler'].state_dict().items() if k != 'step'}, rng=rng(),
            criterion=objects['criterion'].state_dict(),
            criterion_accumulators={k: getattr(objects['criterion'], k) for k in
                ('total_loss', 'reg_loss', 'conf_loss', 'diff_loss', 'loss_cnt', 'loss_dict')
                if hasattr(objects['criterion'], k)})
        atomic_torch(path, state, original_save)
        return sha256(path)

    def create(parameters):
        model = original_create(parameters)
        objects['model'] = model
        # No loader exists on the fresh path. Only our own committed state is permitted on resume.
        if saved_state is not None and completed == 0:
            model.load_state_dict(saved_state['model'], strict=True)
        def epoch_observer(module, inputs):
            objects['current_epoch'] = int(inputs[0]['epoch']) + 1
        model.register_forward_pre_hook(epoch_observer)
        return model

    def create_loss(parameters):
        criterion = original_loss(parameters)
        objects['criterion'] = criterion
        if saved_state is not None:
            criterion.load_state_dict(saved_state['criterion'], strict=True)
            for key, value in saved_state['criterion_accumulators'].items():
                setattr(criterion, key, value)
        def observe_loss(module, inputs, result):
            loss = float(result.detach().item())
            if not math.isfinite(loss):
                raise ValueError('Nonfinite official detection loss')
            if objects['model'].training:
                observations.setdefault(objects['current_epoch'], dict(train_losses=[]))['train_losses'].append(loss)
        criterion.register_forward_hook(observe_loss)
        return criterion

    def optimizer(parameters, model):
        result = original_optimizer(parameters, model)
        objects['optimizer'] = result
        return result

    def load_saved_model(path, model):
        if Path(path) != output or saved_state is None or not completed:
            raise ValueError('External/weights-only initialization is forbidden')
        model.load_state_dict(saved_state['model'], strict=True)
        return completed, model

    def scheduler(parameters, opt, init_epoch=None, n_iter_per_epoch=None):
        # Reconstruct the official scheduler once, then restore its full state instead of its
        # original weights-only catch-up path. The uninterrupted schedule remains byte-for-byte official.
        result = original_scheduler(parameters, opt, n_iter_per_epoch=n_iter_per_epoch)
        objects['scheduler'] = result
        if saved_state is not None:
            opt.load_state_dict(saved_state['optimizer'])
            result.load_state_dict(saved_state['scheduler'])
            restore_rng(saved_state['rng'])
        else:
            digest = save_state(0)
            atomic_json(state_dir / '00.json', dict(contract_identity=contract['identity'], sha256=digest))
        step = result.step
        def finish(epoch=None):
            if epoch is None:
                raise ValueError('Unexpected scheduler step dispatch')
            value = step(epoch)  # Preserve actual official zero-based explicit-epoch behavior.
            # The scheduler stores arbitrary instance attributes in state_dict. Our observer
            # method is deliberately excluded by save_state; it is not experiment state.
            number = epoch + 1
            row = observations[number]
            if not row.get('clean_ap_summary') or not row.get('train_losses'):
                raise ValueError('Epoch missing training/Clean AP evidence')
            digest = save_state(number)
            metrics = dict(status='complete', epoch=number, contract_identity=contract['identity'],
                checkpoint_sha256=sha256(output / f'net_epoch{number}.pth'),
                train_loss=sum(row['train_losses']) / len(row['train_losses']), train_updates=len(row['train_losses']),
                validation_loss=row['validation_loss'],
                clean_ap_summary=row['clean_ap_summary'], clean_ap_summary_sha256=sha256(row['clean_ap_summary']),
                clean_AP=row['clean_AP'], next_learning_rates=[g['lr'] for g in opt.param_groups],
                resume_state_sha256=digest, smoke_only=args.smoke,
                output_hashes={Path(row['clean_ap_checkpoint']).name: row['clean_ap_checkpoint_sha256']})
            atomic_json(output / 'epochs' / f'{number:02d}' / 'complete.json', metrics)
            all_rows = commits + [complete(output / 'epochs' / f'{i:02d}' / 'complete.json') for i in range(completed + 1, number + 1)]
            atomic_json(output / 'history.json', dict(status='running', epochs=all_rows))
            atomic_json(output / 'progress.json', dict(status='running', completed_epoch=number, total_epochs=20))
            print(f'D-1D EPOCH COMMITTED {number}/20: Clean AP={row["clean_AP"]}', flush=True)
            if args.smoke:
                raise SmokeFinished()
            return value
        result.step = finish
        return result

    def save_weights(value, path, *a, **kw):
        file = Path(path)
        match = re.fullmatch(r'net_epoch(\d+)\.pth', file.name)
        if file.parent != output or not match or a or kw:
            raise ValueError('Unexpected official checkpoint save path')
        if int(match[1]) <= completed:
            raise ValueError('Refusing to overwrite a committed checkpoint')
        atomic_torch(file, value, original_save)

    class Writer:
        def __init__(self, *a, **kw):
            self.writer = original_writer(*a, **kw)
            objects['writer'] = self.writer
        def add_scalar(self, tag, value, epoch, *a, **kw):
            result = self.writer.add_scalar(tag, value, epoch, *a, **kw)
            if tag != 'Validate_Loss':
                return result
            number = epoch + 1
            if not math.isfinite(float(value)):
                raise ValueError('Nonfinite validation loss')
            checkpoint = output / f'net_epoch{number}.pth'
            digest = sha256(checkpoint)
            epoch_root = output / 'epochs' / f'{number:02d}'
            epoch_root.mkdir(parents=True, exist_ok=True)
            immutable_weight = epoch_root / ('checkpoint_' + digest + '.pth')
            if not immutable_weight.exists():
                # Same-filesystem hard link preserves even an interrupted epoch's AP input
                # when its uncommitted net_epochN filename is atomically replaced on retry.
                os.link(checkpoint, immutable_weight)
            if sha256(immutable_weight) != digest:
                raise ValueError('Immutable per-attempt AP checkpoint changed')
            destination = epoch_root / ('clean_ap_' + digest)
            command = [sys.executable, '-B', str(HERE / 'main.py'), 'clean-ap', '--checkpoint', str(immutable_weight),
                '--evaluation-root', str(destination), '--resume']
            if args.smoke:
                command += ['--limit-frames', '12']
            for key in DEFAULTS:
                command += ['--' + key.replace('_', '-'), getattr(args, key)]
            # Child inference has its own frozen RNG. It cannot change training RNG/model/optimizer.
            torch.cuda.empty_cache()
            with (output / f'clean_ap_epoch{number:02d}.log').open('a') as log:
                subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            ap = complete(destination / 'summary.json')
            if ap['checkpoint_sha256'] != digest or ap['conditions'] != ['clean'] or (not args.smoke and ap['frame_count'] != 829):
                raise ValueError('Epoch Clean AP evidence differs')
            observations[number].update(validation_loss=float(value), clean_ap_summary=str(destination / 'summary.json'),
                clean_AP=ap['results']['clean']['AP'], clean_ap_checkpoint=str(immutable_weight),
                clean_ap_checkpoint_sha256=digest)
            return result
        def __getattr__(self, key):
            return getattr(self.writer, key)

    original_build = official.build_dataset
    def build(parameters, visualize=False, train=True):
        dataset = original_build(parameters, visualize=visualize, train=train)
        expected = train_manifest['counts']['cooperative_frames'] if train else 829
        if (len(dataset) != expected or dataset.max_cav != 7 or not dataset.fov or not dataset.proj_first
                or not dataset.use_Lidar or dataset.use_Radar or dataset.use_DeLidar):
            raise ValueError('Official train/validation dataset flags/counts differ')
        if train:
            recovered = []
            for index, scene in enumerate(train_manifest['scenes']):
                content = dataset.scenario_database[index]
                if list(content) != scene['agents'][:7] or not content[scene['ego']]['ego']:
                    raise ValueError('Official training ego/CAV order differs from extraction')
                for cav, records in content.items():
                    if [t for t in records if t != 'ego'] != scene['timestamps']:
                        raise ValueError('Official train timestamp order differs')
                    for stamp in scene['timestamps']:
                        if Path(records[stamp]['lidar']) != Path(args.train_root) / 'train' / scene['scene'] / cav / (stamp + '.pcd'):
                            raise ValueError('Weather file selected during Clean training')
                recovered += [[scene['scene'], t] for t in scene['timestamps']]
            if len(recovered) != train_manifest['counts']['cooperative_frames']:
                raise ValueError('Official train cooperative frame count differs')
        return dataset

    def loader(dataset, *a, **kw):
        expected = dict(batch_size=8, num_workers=16, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
        if a or any(kw.get(k) != v for k, v in expected.items()) or set(kw) != set(expected) | {'collate_fn'}:
            raise ValueError('Official training/loss-validation loader behavior changed')
        if args.smoke:
            dataset = torch.utils.data.Subset(dataset, range(16))
        return original_loader(dataset, **kw)

    def suppress_inference(command):
        expected = f'python opencood/tools/inference.py --model_dir {output} --fusion_method intermediate'
        if command != expected:
            raise ValueError('Unexpected automatic shell command: ' + command)
        print('Official automatic inference replaced by explicit Clean-selected D-1C final evaluation command.', flush=True)
        return 0

    def forbid_weight_load(*a, **kw):
        raise ValueError('No external torch.load is permitted inside official training; resume was already strictly restored')

    replacements = [(utils, 'create_model', create), (utils, 'create_loss', create_loss),
        (utils, 'setup_optimizer', optimizer), (utils, 'setup_lr_schedular', scheduler),
        (utils, 'setup_train', lambda h: str(output)), (utils, 'load_saved_model', load_saved_model),
        (official, 'build_dataset', build), (official, 'DataLoader', loader),
        (official, 'SummaryWriter', Writer), (torch, 'save', save_weights),
        (torch, 'load', forbid_weight_load), (os, 'system', suppress_inference)]
    with ExitStack() as stack:
        for obj, name, replacement in replacements:
            stack.enter_context(patched(obj, name, replacement))
        previous = sys.argv
        try:
            sys.argv = [str(entry), '--hypes_yaml', str(config), '--worker', '16', '--tag', 'D-1D_Clean_random']
            if completed:
                sys.argv += ['--model_dir', str(output)]
            official.main()
        except SmokeFinished:
            atomic_json(output / 'SMOKE_ONLY.json', dict(status='complete', official_train_batches=2,
                validation_loss_batches=2, clean_AP_frames=12, eligible_for_model_selection=False))
            print('Training smoke completed. Never use this directory for full model selection.', flush=True)
            return
        finally:
            sys.argv = previous
            if 'writer' in objects:
                objects['writer'].close()
    from evaluation import selection
    chosen = selection(args)
    finish_training(output, contract, chosen, args)
    print('Training complete. Clean-only selection locked:', chosen['epoch'], flush=True)


class SmokeFinished(Exception):
    pass


def verify_clean_loss_view(args, evidence):
    """The loss loader uses D-1A's view; AP uses D-1B's overlay. Verify both views' links."""
    from common import no_test
    view = read_json(Path(args.view_root) / 'manifest.json')
    validated = {str(no_test(r['source']).resolve(strict=True)): r['sha256']
                 for r in evidence['input_manifest']['files']}
    for row in view['files']:
        file = no_test(row['view'])
        source = no_test(row['source']).resolve(strict=True)
        if (file.resolve(strict=True) != source or file.stat().st_size != row['size']
                or validated.get(str(source)) != row['sha256']):
            raise ValueError('Clean validation loss input link/hash differs from D-1A/B: ' + str(file))
    expected = {str(no_test(r['view'])) for r in view['files']}
    if {str(p) for p in (Path(args.view_root) / 'validate').rglob('*') if p.is_file()} != expected:
        raise ValueError('Unexpected/missing files in the Clean validation loss view')


def finish_training(output, contract, chosen, args):
    from observe import module_inventory
    if (output / 'training_summary.json').exists():
        complete(output / 'training_summary.json')
        return
    atomic_json(output / 'protocol.json', dict(**contract,
        imported_modules=module_inventory(Path(args.runtime_root)), selection=chosen))
    rows = [complete(output / 'epochs' / f'{epoch:02d}' / 'complete.json') for epoch in range(1, 21)]
    atomic_json(output / 'history.json', dict(status='complete', epochs=rows))
    atomic_json(output / 'progress.json', dict(status='training_complete', completed_epoch=20,
        selected_epoch=chosen['epoch'], weather_inference_started=False))
    names = ['contract.json', 'config.yaml', 'selection_policy.json', 'selected.json', 'protocol.json', 'history.json']
    names += [f'net_epoch{epoch}.pth' for epoch in range(1, 21)]
    atomic_json(output / 'training_summary.json', dict(status='complete', epochs=20,
        train_data_acceptance=contract['train_data_acceptance'],
        selected_epoch=chosen['epoch'], initialization='random', weather_used_for_selection=False,
        output_hashes={name: sha256(output / name) for name in names}))
