"""Server-only host compiler/header probe; no dataset or detector execution."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from common import atomic_json, server_only, sha256


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = dict(status='checking', python=sys.executable,
        environment={k: os.environ.get(k) for k in
            ('CC', 'CXX', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'ROCM_HOME', 'HIP_HOME')})
    try:
        import torch
        from torch.utils.cpp_extension import include_paths
        report.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      torch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise RuntimeError(f'CXX compiler not found: {compiler}')
        report['compiler_command'] = compiler
        version = subprocess.run(compiler + ['--version'], text=True, capture_output=True, check=True)
        report['compiler_version'] = version.stdout
        macros = subprocess.run(compiler + ['-dM', '-E', '-x', 'c++', '-'],
            input='', text=True, capture_output=True, check=True)
        defines = dict(re.findall(r'^#define\s+(\w+)\s+(\S+)', macros.stdout, re.MULTILINE))
        family = 'clang' if '__clang__' in defines else 'gcc' if '__GNUC__' in defines else 'unknown'
        major = int(defines.get('__clang_major__' if family == 'clang' else '__GNUC__', '0'))
        report.update(compiler_family=family, compiler_major=major)
        includes = include_paths()
        candidates = list(dict.fromkeys(includes + [str(Path(sys.prefix) / 'include'),
            '/usr/include', '/usr/local/include', '/opt/dtk-25.04.2/include'] +
            [p for k in ('CPATH', 'CPLUS_INCLUDE_PATH') for p in os.environ.get(k, '').split(':') if p]))
        report['headers'] = {}
        for name in ('gflags/gflags.h', 'glog/stl_logging.h', 'glog/logging.h'):
            found = [Path(p) / name for p in candidates if (Path(p) / name).is_file()]
            report['headers'][name] = [dict(path=str(p), sha256=sha256(p)) for p in found]
        if family not in ('gcc', 'clang') or major < 9:
            raise RuntimeError(f'Host {family} major version {major} is too old; '
                               'PyTorch headers require GCC/Clang >=9. Set CC/CXX explicitly.')
        command = compiler + ['-std=c++17', '-fsyntax-only', '-x', 'c++', '-',
            '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
        command += ['-I' + path for path in includes]
        report['syntax_probe_command'] = command
        source = '#include <c10/util/C++17.h>\n#include <c10/util/Flags.h>\n#include <c10/util/Logging.h>\n'
        result = subprocess.run(command, input=source, text=True, capture_output=True)
        report.update(probe_stdout=result.stdout, probe_stderr=result.stderr, probe_returncode=result.returncode)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError('PyTorch C++ header probe failed; see build_environment.json before rebuilding extensions')
        report['status'] = 'complete'
        print(f'Build preflight passed: {family} {major}; PyTorch host headers compile.', flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
