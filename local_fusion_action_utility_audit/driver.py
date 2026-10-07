"""Pipe the pipeline's combined output to the console and driver.log once."""
import subprocess
import signal
import sys
from .common import validate_run


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    arguments = sys.argv[1:]
    if '--run' not in arguments:
        raise ValueError('--run is required')
    run = validate_run(arguments[arguments.index('--run') + 1])
    with (run / 'driver.log').open('a', encoding='utf-8', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'local_fusion_action_utility_audit.pipeline',
                                  *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        try:
            for line in child.stdout:
                log.write(line)
                # launch.sh already redirects this process to driver.log; avoid double writes.
                if not __import__('os').environ.get('AUDIT_LAUNCH_REDIRECT'):
                    print(line, end='', flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
