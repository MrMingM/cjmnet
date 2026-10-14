"""Server entry point: assets -> checks -> generation -> pure AttFuse evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from reference_weather.common import (PACKAGE, external_output, load_config,
                                     server_only, validate_paths, verify_sources)


@contextmanager
def exclusive_run(output):
    import fcntl
    path = Path(output).with_name(Path(output).name + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is using {output}') from error
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'assets', 'generate', 'evaluate', 'report'], default='all')
    parser.add_argument('--config', default=str(PACKAGE / 'config.yaml'))
    parser.add_argument('--output-root')
    parser.add_argument('--assets-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit-frames', type=int, default=0, help='Smoke only; 0 means all frames.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    server_only()
    verify_sources()
    config = load_config(args.config)
    for name in ('output_root', 'assets_root', 'checkpoint', 'model_config'):
        if getattr(args, name) is not None:
            config[name] = str(external_output(getattr(args, name))) if name.endswith('_root') else getattr(args, name)
    if args.workers is not None:
        config['generation_workers'] = args.workers
    validate_paths(config)
    if args.limit_frames < 0 or int(config['generation_workers']) < 1:
        parser.error('Frame limit must be nonnegative and workers positive')
    with exclusive_run(config['output_root']):
        if args.stage == 'all':
            from reference_weather.preflight import preflight
            preflight(config, require_gpu=True)
        if args.stage in ('all', 'generate'):
            subprocess.run([sys.executable, '-m', 'unittest', 'reference_weather.test_contract'], check=True)
        if args.stage in ('all', 'assets'):
            from reference_weather.prepare_assets import prepare
            prepare(config)
        if args.stage in ('all', 'generate'):
            from reference_weather.generate import generate
            generate(config, limit_frames=args.limit_frames, resume=args.resume)
        if args.stage in ('all', 'evaluate'):
            from reference_weather.evaluate import evaluate
            evaluate(config, limit_frames=args.limit_frames, resume=args.resume)
        if args.stage == 'report':
            from reference_weather.evaluate import report
            from reference_weather.generate import verify_dataset
            report(config, manifest=verify_dataset(config, args.limit_frames))


if __name__ == '__main__':
    main()
