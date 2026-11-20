"""Run CLASSY and GRANDE, which need environments of their own, as subprocesses of a user-chosen interpreter.

    FLARE_CLASSY_PYTHON=/path/to/classy-env/bin/python    FLARE_GRANDE_PYTHON=/path/to/grande-env/bin/python

The environments are described in baselines/README.md. The scripts (classy.py, grande.py) read a fold from an .npz file
and write their predictions to another one, so nothing but numpy arrays crosses the boundary.
"""
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INTERPRETER = {'CLASSY': 'FLARE_CLASSY_PYTHON', 'GRANDE': 'FLARE_GRANDE_PYTHON'}
SCRIPT = {'CLASSY': 'classy.py', 'GRANDE': 'grande.py'}


def interpreter(name):
    """The Python of the environment of `name`, or None when its variable is not set."""
    return os.environ.get(INTERPRETER[name]) or None


def run(name, directory, arrays, **options):
    """Run the worker of `name` on `arrays` (saved to in.npz) with `--key value` options; the contents of out.npz."""
    python = interpreter(name)
    if python is None:
        raise RuntimeError(f'{name} needs its own environment: set {INTERPRETER[name]} (see baselines/README.md)')
    np.savez(os.path.join(directory, 'in.npz'), **arrays)
    command = [python, '-u', os.path.join(HERE, SCRIPT[name]), '--dir', directory]
    for key, value in options.items():
        command += [f'--{key}', str(value)]
    result = subprocess.run(command, capture_output=True, text=True)
    out = os.path.join(directory, 'out.npz')
    if result.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f'{name} worker failed: {result.stderr[-600:]}')
    return np.load(out, allow_pickle=True)
