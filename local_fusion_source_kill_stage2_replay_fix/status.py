"""Read-only status across the original K1 and repaired K2 runs."""
import argparse
from pathlib import Path
from local_fusion_source_kill_stage2.common import read_json, safe_path


def status(run):
    run = safe_path(run)
    protocol = read_json(run/'PROTOCOL.json') if (run/'PROTOCOL.json').exists() else None
    parent = Path(protocol['k2_replay_repair']['parent_run']) if protocol else None
    def state(path):
        return read_json(path).get('status', 'pending') if path and path.exists() else 'pending'
    print('K1_PARENT = '+state(parent/'k1/manifest.json' if parent else None))
    print('SOURCE_AP_GATE = '+state(run/'source_reproduction/manifest.json'))
    print('K2_REPAIR = '+state(run/'k2/manifest.json'))
    print('REPAIR_DRIVER = '+state(run/'repair_manifest.json'))
    print('STAGE2 = '+state(run/'STAGE2_FINAL.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    status(parser.parse_args().run)


if __name__ == '__main__':
    main()
