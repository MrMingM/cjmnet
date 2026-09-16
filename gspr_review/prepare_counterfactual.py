"""Create explicit matched configs in a new experiment directory; never reuse runs."""
import argparse
from pathlib import Path
import yaml
from .runtime import new_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-config', default='gspr_review/experiment.yaml')
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base_config).read_text(encoding='utf-8'))
    out = new_output(args.output_dir)
    for architecture in ('legacy', 'contrast', 'contrast_gain'):
        options = dict(base, review=dict(base['review'], architecture=architecture))
        (out/(architecture+'.yaml')).write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__':
    main()
