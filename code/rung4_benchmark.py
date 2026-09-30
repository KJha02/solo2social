"""Pinned LiveCodeBench tasks and container-only execution; no model dependency.

Training feedback uses official tests. Repeated held-out tasks never enter the
revision feedback. The agent's tool container receives only its own workspace;
the scorer runs separately with the official tests and returns aggregate scores.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import tempfile
import time
import zlib

ROOT = Path(os.environ.get('SOLO2SOCIAL_DEPS', Path(__file__).resolve().parents[1] / 'data/dependencies'))
OFFICIAL_ROOT = ROOT / 'LiveCodeBench'
RUNTIME = ROOT / 'lcb_runtime'
LCB_COMMIT = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'
DATA_REVISION = '0fe84c3912ea0c4d4a78037083943e8f0c4dd505'


class _StringUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Dataset pickle must contain data only')


def _tests(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(value))
        return json.loads(_StringUnpickler(io.BytesIO(raw)).load())


def prepare_dataset(output, train_count=32, holdout_count=16, split_seed=0,
                    source=RUNTIME / 'test6.jsonl'):
    """Deterministically sample the release-v6 increment, not all LCB.

    Download the pinned test6.jsonl before calling. Split before model runs and
    preserve the manifest. This is a declared subset, not the leaderboard suite.
    """
    source, output = Path(source), Path(output)
    rows = [json.loads(line) for line in source.open() if line.strip()]
    rows.sort(key=lambda x: (x['platform'], str(x['question_id'])))
    random.Random(split_seed).shuffle(rows)
    if train_count + holdout_count > len(rows):
        raise ValueError(f'Requested too many tasks; only {len(rows)} in increment')
    tasks = []
    for index, row in enumerate(rows[:train_count + holdout_count]):
        row['id'] = f"{row['platform']}:{row['question_id']}"
        row['split'] = 'train' if index < train_count else 'holdout'
        row['public_test_cases'] = _tests(row['public_test_cases'])
        row['private_test_cases'] = _tests(row['private_test_cases'])
        row['metadata'] = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
        tasks.append(row)
    manifest = dict(dataset='livecodebench/code_generation_lite', revision=DATA_REVISION,
                    subset='release_v6 increment test6.jsonl', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    official_commit=LCB_COMMIT, split_seed=split_seed,
                    train_count=train_count, holdout_count=holdout_count, tasks=tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise FileExistsError('Refusing to replace a different benchmark split')
    output.write_text(json.dumps(manifest))
    return manifest


class LiveCodeBench:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text())
        self.tasks = {x['id']: x for x in self.manifest['tasks']}
        self.train_ids = [k for k, v in self.tasks.items() if v['split'] == 'train']
        self.holdout_ids = [k for k, v in self.tasks.items() if v['split'] == 'holdout']

    def task(self, task_id):
        return self.tasks[task_id]

    def public_task(self, task_id):
        row = self.task(task_id)
        return {k: row[k] for k in ('id', 'question_title', 'question_content',
                                    'starter_code', 'public_test_cases', 'platform')}


class Sandbox:
    def __init__(self, image=RUNTIME / 'python311.sif'):
        self.image = Path(image).resolve()

    def run(self, workspace, command, timeout=10, *, _mounts=(), _env=()):
        """Run a shell command with no network and only /work persisted.

        Mount/env extensions are scorer-internal, never agent tool arguments.
        Container setup failure raises instead of counting as model failure.
        """
        workspace = Path(workspace).resolve()
        if not workspace.is_dir() or not self.image.is_file():
            raise FileNotFoundError('Missing workspace or sandbox image')
        argv = ['apptainer', 'exec', '--containall', '--cleanenv', '--net', '--network', 'none',
                '--no-mount', 'hostfs,cwd,home,bind-paths', '--pwd', '/work',
                '--bind', f'{workspace}:/work']
        # Identical installed libraries for public tools and hidden scoring.
        for host, target in ((RUNTIME / 'python', '/deps'),) + tuple(_mounts):
            argv += ['--bind', f'{Path(host).resolve()}:{target}:ro']
        for key, value in (('PYTHONPATH', '/deps'), ('OPENBLAS_NUM_THREADS', '1')) + tuple(_env):
            argv += ['--env', f'{key}={value}']
        argv += [str(self.image), '/bin/sh', '-c',
                 'ulimit -t 60; ulimit -f 16384; ulimit -v 2097152; ' + command]
        # Remove ambient bind/environment injection into the container.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('APPTAINER', 'SINGULARITY'))}
        started = time.monotonic()
        # A file avoids unbounded capture of generated output in host memory.
        with tempfile.TemporaryFile() as capture:
            process = subprocess.Popen(argv, cwd=workspace, env=env, stdout=capture,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            capture.seek(0)
            output = capture.read(16000).decode(errors='replace')
        if 'FATAL:' in output:
            raise RuntimeError(f'Container infrastructure failure: {output}')
        return dict(returncode=process.returncode, output=output, timed_out=timed_out,
                    seconds=time.monotonic() - started)


def score(task, solution_path, sandbox, official_root=OFFICIAL_ROOT, timeout=6):
    """Official LCB test semantics, including functional and stdin tasks.

    Aggregate only: do not return hidden inputs, outputs, or scorer diagnostics.
    The ephemeral scoring workspace is never exposed to an agent tool call.
    Fixtures are unlinked before running submitted code. The official checker
    nevertheless executes code in-process with test data in memory: it is not
    an adversarially secure grader against Python introspection. The container
    protects the host; it does not change this upstream evaluation limitation.
    """
    tests = task['public_test_cases'] + task['private_test_cases']
    sample = {'input_output': json.dumps(dict(inputs=[t['input'] for t in tests],
               outputs=[t['output'] for t in tests], fn_name=task['metadata'].get('func_name')))}
    with tempfile.TemporaryDirectory(prefix='lcb-score-') as directory:
        work = Path(directory)
        (work / 'solution.py').write_text(Path(solution_path).read_text())
        (work / 'sample.json').write_text(json.dumps(sample))
        (work / 'score.py').write_text(
            'import json, importlib.util, contextlib, io, os\n'
            's=importlib.util.spec_from_file_location("official_testing", "/official/lcb_runner/evaluation/testing_util.py")\n'
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
            'with open("sample.json") as f: sample=json.load(f)\n'
            'os.unlink("sample.json")\n'
            'with open("solution.py") as f: solution=f.read()\n'
            'with contextlib.redirect_stdout(io.StringIO()):\n'
            f' r, meta=m.run_test(sample,test=solution,debug=False,timeout={int(timeout)})\n'
            'passed=sum(bool(x == True) for x in r)\n'
            'print("LCB_RESULT="+json.dumps({"passed_tests":passed,"evaluated_tests":len(r)}))\n')
        result = sandbox.run(work, 'python3 score.py', timeout=min(180, (timeout + 1) * len(tests) + 10),
                             _mounts=((official_root, '/official'),))
    parsed = next((json.loads(x[len('LCB_RESULT='):]) for x in result['output'].splitlines()
                   if x.startswith('LCB_RESULT=')), None)
    if parsed is None and not result['timed_out']:
        raise RuntimeError(f'Official scorer failed: {result["output"][:1000]}')
    passed = parsed['passed_tests'] if parsed else 0
    return dict(pass_at_1=bool(tests) and passed == len(tests),
                reward=float(bool(tests) and passed == len(tests)),
                # Official checker stops at the first failure. This is a lower
                # bound, not an unbiased partial-credit reward.
                passed_test_fraction_lower_bound=passed / len(tests) if tests else 0.,
                evaluated_tests=parsed['evaluated_tests'] if parsed else 0,
                passed_tests=passed, total_tests=len(tests),
                timed_out=result['timed_out'], seconds=result['seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--holdout-count', type=int, default=16)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare_dataset(args.prepare, args.train_count, args.holdout_count)
        print(json.dumps({k: v for k, v in manifest.items() if k != 'tasks'}))
    if args.diagnostic:
        sandbox = Sandbox()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            sentinel = work.parent / f'lcb-host-only-{os.getpid()}'
            sentinel.write_text('not mounted')
            try:
                command = ('python3 -c "import os, socket, numpy; '
                           'assert numpy.__version__ == \'2.2.6\'; '
                           f'assert not os.path.exists(\'{sentinel}\'); '
                           'assert socket.if_nameindex() == [(1, \'lo\')]; print(\'isolated\')"')
                isolated = sandbox.run(work, command)
                assert isolated['returncode'] == 0, isolated
            finally:
                sentinel.unlink()
            skills = work / 'skills'
            skills.mkdir()
            readonly = sandbox.run(work, 'if touch skills/forbidden; then exit 1; fi',
                                   _mounts=((skills, '/work/skills'),))
            assert readonly['returncode'] == 0 and not (skills / 'forbidden').exists(), readonly
            task = dict(public_test_cases=[dict(input='2 3\n', output='5\n')],
                        private_test_cases=[dict(input='4 9\n', output='13\n')], metadata={})
            solution = work / 'answer.py'
            solution.write_text('import os, numpy\nassert numpy.__version__ == "2.2.6"\nassert not os.path.exists("sample.json")\nprint(sum(map(int,input().split())))')
            correct = score(task, solution, sandbox)
            assert correct['pass_at_1'], correct
            solution.write_text('print(0)')
            incorrect = score(task, solution, sandbox)
            assert not incorrect['pass_at_1'], incorrect
            task = dict(public_test_cases=[dict(input='2\n3', output='5')],
                        private_test_cases=[], metadata={'func_name': 'add'})
            solution.write_text('class Solution:\n def add(self,a,b): return a+b\n')
            functional = score(task, solution, sandbox)
            assert functional['pass_at_1'], functional
            print(json.dumps(dict(isolation=isolated, readonly=readonly, correct=correct,
                                  incorrect=incorrect, functional=functional)))
