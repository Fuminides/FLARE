"""The authors' public code that some baselines call, and where it is looked for.

The repositories are not part of this repository: `fetch_external.py` clones them at the pinned commits recorded here
into `baselines/external/` (git-ignored), or into the directory in `$FLARE_EXTERNAL_DIR`. Each adapter asks `repo_path`
for its clone and fails with an instruction when it is missing.
"""
import os
import sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (clone URL, commit, licence file in the clone or None)
REPOS = {
    'rrl': ('https://github.com/12wang3/rrl.git', 'f8d0886b23c4e15f63c62c248b97d4eb73386ad1', 'LICENSE'),
    'RLNet': ('https://github.com/luciledierckx/RLNet.git', '2e7ad9504ff93f7e6c5320ddfa222ed1d8779289', 'LICENCE'),
    'DRNet': ('https://github.com/Joeyonng/decision-rules-network.git', 'ac246983407fbb78b18369f6936ff0c1657accc3', 'LICENSE'),
    'neurules': ('https://github.com/Schascha1/neurules.git', 'b7776cee9d5a6e74fd9e3a2c96e37265549a1cf3', None),
}


def external_dir():
    return os.environ.get('FLARE_EXTERNAL_DIR') or os.path.join(HERE, 'external')


def repo_path(name):
    """Directory of the clone of `name`; ImportError (with the command that fixes it) when it is not there."""
    path = os.path.join(external_dir(), name)
    if not os.path.isdir(path):
        raise ImportError(f'the code of {name} ({REPOS[name][0]}) is not in {external_dir()}: run '
                          f'`python -m baselines.fetch_external`')
    return path


@contextmanager
def prepend_sys_path(path):
    """Make the modules of a clone importable (they are scripts, not packages) and restore sys.path afterwards."""
    saved = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = saved
