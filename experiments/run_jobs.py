"""Run a job list (one shell command per line, see experiments/jobs.py) on this machine.

    python -m experiments.run_jobs results/jobs/flare.txt --workers 4
    python -m experiments.run_jobs results/jobs/baselines_cc18.txt --workers 16 --lines 1-500

Each line runs in its own process from the repository root; its output goes to <list folder>/logs/<list>_<line>.log.
A line that fails is reported and the others go on (the commands resume, so running the list again retries only what
is missing). Every process gets OMP_NUM_THREADS=1 unless it is set. One progress line "[k/N] ..." is printed per
finished command.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('list')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lines', default=None, help='a range of lines, 1-based and inclusive, e.g. 1-100')
    args = parser.parse_args(argv)
    with open(args.list) as f:
        lines = [(i, line.strip()) for i, line in enumerate(f, 1) if line.strip()]
    if args.lines:
        first, last = (int(v) for v in args.lines.split('-'))
        lines = [(i, line) for i, line in lines if first <= i <= last]
    logs = os.path.join(os.path.dirname(os.path.abspath(args.list)), 'logs')
    os.makedirs(logs, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.list))[0]
    env = {**os.environ, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '1')}

    def run(index, command):
        with open(os.path.join(logs, f'{name}_{index}.log'), 'w') as log:
            # the lines start with "python": run them with this interpreter
            command = command.replace('python ', f'{sys.executable} -u ', 1) if command.startswith('python ') else command
            return subprocess.call(command, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, i, line): (i, line) for i, line in lines}
        for done, future in enumerate(as_completed(futures), 1):
            index, line = futures[future]
            code = future.result()
            failed += code != 0
            print(f'[{done}/{len(lines)}] line {index} exit={code}', flush=True)
    print(f'{len(lines) - failed} lines finished, {failed} failed (logs in {logs})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
