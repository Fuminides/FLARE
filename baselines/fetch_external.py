"""Clone the authors' code of the baselines that need it (RRL, RL-Net, DR-Net, NeuRules) at the pinned commits.

    python -m baselines.fetch_external [--dir DIR]

DIR defaults to `$FLARE_EXTERNAL_DIR`, else `baselines/external/`. An existing clone is moved to the pinned commit.
The licence of each clone is printed. The NeuRules repository has no licence file, so it is only cloned, never
redistributed here.
"""
import argparse
import os
import subprocess

from baselines.external import REPOS, external_dir


def git(*args, cwd=None):
    subprocess.run(['git', *args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default=None, help='where to clone (default: see above)')
    args = parser.parse_args()
    if args.dir:
        os.environ['FLARE_EXTERNAL_DIR'] = args.dir
    root = external_dir()
    os.makedirs(root, exist_ok=True)
    for name, (url, commit, licence) in REPOS.items():
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            git('clone', '--quiet', url, path)
        else:
            git('fetch', '--quiet', 'origin', cwd=path)
        git('checkout', '--quiet', commit, cwd=path)
        if licence is not None and os.path.exists(os.path.join(path, licence)):
            with open(os.path.join(path, licence)) as f:
                title = f.readline().strip()
            print(f'{name}: {url} @ {commit[:8]}  licence: {title} ({licence})')
        else:
            print(f'{name}: {url} @ {commit[:8]}  licence: NO LICENCE FILE in the repository (do not redistribute it)')


if __name__ == '__main__':
    main()
