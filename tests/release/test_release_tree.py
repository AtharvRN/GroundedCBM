"""Keep the release executable tree independent of legacy experiment modules."""
import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_no_legacy_executable_paths():
    assert not list((ROOT/'gcbm').glob('imagenet_*.py'))
    assert not (ROOT/'gcbm/train_imagenet.py').exists()
    assert not (ROOT/'train_cbm.py').exists()
    for folder in ('scripts','methods','model','data','datasets','evaluations','clip'):
        assert not list((ROOT/folder).rglob('*.py')), folder
    assert not list((ROOT/'configs').glob('*.yaml'))
    assert not list((ROOT/'configs').glob('*.json'))


def test_all_release_internal_imports_resolve_to_source():
    for package in ('gcbm','glm_saga'):
        for path in (ROOT/package).rglob('*.py'):
            module='.'.join(path.relative_to(ROOT).with_suffix('').parts)
            parent=module.rsplit('.',1)[0]
            for node in ast.walk(ast.parse(path.read_text())):
                imports=[]
                if isinstance(node,ast.Import):
                    imports=[alias.name for alias in node.names]
                elif isinstance(node,ast.ImportFrom):
                    name=node.module or ''
                    imports=[importlib.util.resolve_name('.'*node.level+name,parent)
                             if node.level else name]
                for name in imports:
                    if name.split('.')[0] in ('gcbm','glm_saga'):
                        target=ROOT.joinpath(*name.split('.'))
                        assert target.with_suffix('.py').is_file() or target.is_dir(), (path,name)
                    assert name.split('.')[0] not in ('methods','model','data','evaluations','clip'), (path,name)
