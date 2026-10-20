"""Pinned UECP modules live in a private namespace, never in the runtime's opencood namespace."""
import ast
import importlib
import importlib.abc
import importlib.util
import sys

from .common import ROOT, read, sha, HERE

PREFIX = 'uecp_gspr_v2xr.uecp_vendor'
SOURCE = ROOT / 'related_code/UECP-main/opencood'


class Imports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and (node.module == 'opencood' or node.module.startswith('opencood.')):
            node.module = PREFIX + node.module[len('opencood'):]
        return node

    def visit_Import(self, node):
        items = []
        for alias in node.names:
            if alias.name == 'opencood' or alias.name.startswith('opencood.'):
                mapped = PREFIX + alias.name[len('opencood'):]
                if alias.asname:
                    items.append(ast.Import(names=[ast.alias(name=mapped, asname=alias.asname)]))
                else:
                    items.extend([ast.Import(names=[ast.alias(name=mapped, asname='_loaded_uecp_module')]),
                                  ast.Import(names=[ast.alias(name=PREFIX, asname='opencood')])])
            else:
                items.append(ast.Import(names=[alias]))
        return items


class Loader(importlib.abc.Loader):
    def __init__(self, path, package):
        self.path, self.package = path, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.package:
            module.__path__ = [str(self.path)]
            return  # Empty package shells avoid executing unrelated dataset factories.
        rel = self.path.relative_to(ROOT).as_posix()
        if sha(self.path) != read(HERE / 'PINS.json')['files'].get(rel):
            raise ValueError('Unpinned UECP source import: ' + rel)
        tree = Imports().visit(ast.parse(self.path.read_text(encoding='utf-8')))
        module.__file__ = str(self.path)
        exec(compile(ast.fix_missing_locations(tree), str(self.path), 'exec'), module.__dict__)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PREFIX and not fullname.startswith(PREFIX + '.'):
            return None
        tail = fullname[len(PREFIX):].lstrip('.').split('.') if fullname != PREFIX else []
        p = SOURCE.joinpath(*tail)
        is_package = p.is_dir()
        if not is_package:
            p = p.with_suffix('.py')
        if not p.exists():
            raise ImportError('Missing isolated UECP module: ' + fullname)
        return importlib.util.spec_from_loader(fullname, Loader(p, is_package), is_package=is_package)


def module(name):
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    return importlib.import_module(PREFIX + '.' + name)


def inventory():
    return {name: str(getattr(m, '__file__', 'namespace')) for name, m in sys.modules.items()
            if name == PREFIX or name.startswith(PREFIX + '.')}
