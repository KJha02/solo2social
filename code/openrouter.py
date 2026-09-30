"""Small OpenRouter transport with resumable asynchronous batch requests."""

from __future__ import annotations

import hashlib
import http.client
from contextlib import contextmanager
import json
import math
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Any


TERMINAL_BATCH_STATES = {"completed", "failed", "cancelled", "expired"}
_LOCAL_LEDGER_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LEDGER_GUARD = threading.Lock()


class SpendingLimitError(RuntimeError):
    def __init__(self, message: str, pending: bool):
        super().__init__(message)
        self.pending = pending

    def __reduce__(self):
        return (type(self), (str(self), self.pending))


class ProviderResponseError(RuntimeError):
    """An HTTP-success envelope containing a provider failure, not an answer."""

    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(f'OpenRouter provider response failed ({code}): {message}')
        self.retryable = str(code) in {'408', '429', '500', '502', '503', '504'}

    def __reduce__(self):
        return (type(self), (self.code, self.message))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A different temporary name per writer is essential on shared filesystems.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _slurm_job_live(job: str) -> bool | None:
    """Return authoritative liveness when Slurm is available, otherwise unknown."""
    try:
        result = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if 'Invalid job id specified' in result.stderr:
        return False
    return None


def _lock_owner_live(owner: dict) -> bool | None:
    # Slurm preserves the job ID across preemption/requeue. A lock from an older
    # restart belongs to a terminated allocation even when that job ID is live
    # again (possibly on another node). Do not mistake it for a current writer.
    if (owner.get('job') == os.environ.get('SLURM_JOB_ID')
            and owner.get('job') and 'restart_count' in owner):
        current_restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        if current_restart > int(owner['restart_count']):
            return False
    if owner.get('host') == socket.gethostname() and isinstance(owner.get('pid'), int):
        try:
            os.kill(owner['pid'], 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    if owner.get('job'):
        return _slurm_job_live(str(owner['job']))
    return None


def _recover_terminal_owner(directory: Path, stale_after: float) -> bool:
    """Atomically retire a stale lock only when its recorded owner is dead."""
    owner_path = directory / 'owner.json'
    try:
        owner = json.loads(owner_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if time.time() - float(owner.get('created_at', time.time())) < stale_after:
        return False
    if _lock_owner_live(owner) is not False:
        return False
    retired = directory.with_name(directory.name + '.retired-' + uuid.uuid4().hex)
    try:
        directory.rename(retired)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    unexpected = [entry for entry in retired.iterdir() if entry.name != 'owner.json']
    if unexpected:
        raise RuntimeError(f'Refusing to clear ledger lock with unexpected contents: {retired}')
    (retired / 'owner.json').unlink(missing_ok=True)
    retired.rmdir()
    return True


@contextmanager
def _ledger_lock(path: Path, timeout: float = 120, stale_after: float = 30):
    # Queue threads before competing for the filesystem lock. Otherwise a worker
    # can starve for the entire timeout while other local workers keep winning.
    with _LOCAL_LEDGER_GUARD:
        local = _LOCAL_LEDGER_LOCKS.setdefault(str(path.resolve()), threading.Lock())
    with local:
        with _cross_process_ledger_lock(path, timeout, stale_after):
            yield


@contextmanager
def _cross_process_ledger_lock(path: Path, timeout: float, stale_after: float):
    """Acquire a cross-node lock and recover only verified-dead Slurm owners."""
    directory = path.with_suffix('.lockdir')
    deadline = time.monotonic() + timeout
    next_recovery_check = 0.
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            now = time.monotonic()
            if now >= next_recovery_check:
                if _recover_terminal_owner(directory, stale_after):
                    continue
                next_recovery_check = now + 2
            if now >= deadline:
                raise RuntimeError(f'Ledger lock timed out: {directory}; inspect owner.json and Slurm before clearing')
            time.sleep(.05)
    try:
        _atomic_json(directory / 'owner.json', dict(host=socket.gethostname(), pid=os.getpid(),
                     job=os.environ.get('SLURM_JOB_ID'),
                     restart_count=int(os.environ.get('SLURM_RESTART_COUNT', '0')),
                     created_at=time.time()))
        yield
    finally:
        (directory / 'owner.json').unlink(missing_ok=True)
        directory.rmdir()


class OpenRouterClient:
    """OpenAI-chat compatible calls without storing credentials or prompts."""

    def __init__(self, config: dict[str, Any], cache_dir: Path | None = None):
        self.config = dict(config)
        # Operational recovery ceilings do not alter prompts or cache identities.
        for field, env in [('run_cost_limit_usd', 'AGENT_MARKET_RUN_COST_LIMIT_USD'),
                           ('total_cost_limit_usd', 'AGENT_MARKET_TOTAL_COST_LIMIT_USD')]:
            if os.environ.get(env):
                limit = float(os.environ[env])
                if not math.isfinite(limit) or limit <= 0:
                    raise ValueError(f'Invalid spending ceiling: {env}')
                self.config[field] = limit
        key_env = config.get("api_key_env", "OPENROUTER_KEY")
        self.api_key = os.environ.get(key_env)
        if not self.api_key:
            raise RuntimeError(f"OpenRouter credential environment variable is unset: {key_env}")
        self.api_root = config.get("base_url", "https://openrouter.ai/api").rstrip("/")
        if self.api_root.endswith("/v1"):
            self.api_root = self.api_root[:-3]
        declared_model = config["name"]
        self.batch = bool(config.get("batch", declared_model.endswith(":batch")))
        self.model = declared_model.removesuffix(":batch")
        self.poll_seconds = max(2, int(config.get("batch_poll_seconds", 10)))
        self.timeout = int(config.get("timeout", 600))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.ledger = Path(config.get('usage_ledger', os.environ.get('AGENT_MARKET_USAGE_LEDGER', Path(__file__).resolve().parents[1] / 'runs/openrouter_usage/ledger.json')))
        self.run_id = str(self.cache_dir.resolve().parent) if self.cache_dir else 'unassigned'
        self.pricing = config.get('pricing')
        self._pricing_lock = threading.Lock()
        self.phase = 'decision'

    def _account(self, key: str, **updates) -> dict:
        """Cross-process accounting survives response-cache cleanup and failed rounds."""
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with _ledger_lock(self.ledger):
            data = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
            if updates.pop('reserve_new', False) and data.get(key, {}).get('status') is None:
                def exposure(rows):
                    return sum(max(x.get('reserved_usd', 0), x.get('cost_usd') or 0)
                               if x.get('cost_usd') is None else x['cost_usd'] for x in rows)
                proposed = updates['reserved_usd']
                def may_settle(rows, limit):
                    # Unknown/ambiguous charges remain reserved. Only requests
                    # still in flight can free admission capacity by completing.
                    rows = list(rows)
                    pending = [x for x in rows if x.get('status') == 'submitting']
                    settled = [x for x in rows if x.get('status') != 'submitting']
                    return bool(pending) and exposure(settled) + proposed <= limit
                if exposure(data.values()) + proposed > float(self.config.get('total_cost_limit_usd', 40)):
                    raise SpendingLimitError('OpenRouter shared spending limit reached',
                        may_settle(data.values(), float(self.config.get('total_cost_limit_usd', 40))))
                if exposure(x for x in data.values() if x.get('run') == self.run_id) + proposed > float(self.config.get('run_cost_limit_usd', 10)):
                    raise SpendingLimitError('OpenRouter per-run spending limit reached',
                        may_settle((x for x in data.values() if x.get('run') == self.run_id),
                                   float(self.config.get('run_cost_limit_usd', 10))))
            row = data.setdefault(key, {'run': self.run_id, 'model': self.model,
                                        'created_at': time.time(), 'cost_usd': None})
            row.update(updates)
            row['run_cost_limit_usd'] = float(self.config.get('run_cost_limit_usd', 10))
            row['total_cost_limit_usd'] = float(self.config.get('total_cost_limit_usd', 40))
            row['updated_at'] = time.time()
            _atomic_json(self.ledger, data)
            return dict(row)

    def _reserve(self, key: str, bodies: list[dict]) -> None:
        if self.cache_dir is None:
            raise ValueError('OpenRouter requires a per-run cache directory for cost accounting')
        if self.pricing is None:
            # A parallel evaluation should fetch pricing once, not once per
            # worker racing through the first reservation.
            with self._pricing_lock:
                if self.pricing is None:
                    models = self._json('GET', self.api_root + '/v1/models')['data']
                    self.pricing = next(x['pricing'] for x in models if x['id'] == self.model)
        ceiling = self.config.get('request_options', {}).get('provider', {}).get('max_price', {})
        rates = {field: max(float(self.pricing[field]), float(ceiling.get(field, 0)) / 1e6)
                 for field in ('prompt', 'completion')}
        # Reserve undiscounted text pricing, with one token per UTF-8 byte plus
        # framing overhead and 2x output headroom. This is a guard, not a bill.
        estimate = sum((len(json.dumps(b).encode()) + 4096) * rates['prompt']
                       + 2 * int(b['max_tokens']) * rates['completion'] for b in bodies)
        if not math.isfinite(estimate) or estimate <= 0:
            raise RuntimeError('Cannot bound OpenRouter request cost from model pricing')
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._account(key, reserve_new=True, reserved_usd=estimate, status='submitting',
                              request_count=len(bodies), pricing=self.pricing, phase=self.phase)
                return
            except SpendingLimitError as exc:
                if not exc.pending or time.monotonic() >= deadline:
                    raise
                time.sleep(.5)

    def reconcile_generation_costs(self) -> dict:
        """Resolve known generation IDs; never release unidentified charges."""
        from concurrent.futures import ThreadPoolExecutor
        with _ledger_lock(self.ledger):
            records = json.loads(self.ledger.read_text())
        pending = [(key, row['generation_id']) for key, row in records.items()
                   if row.get('cost_usd') is None and row.get('generation_id')]
        def lookup(item):
            key, generation = item
            try:
                data = self._json('GET', self.api_root + '/v1/generation?' +
                                  urllib.parse.urlencode({'id': generation}))['data']
                value = data.get('total_cost')
                if (data.get('id') != generation or value is None or
                        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0):
                    return None
                return key, generation, float(value)
            except (RuntimeError, OSError, ValueError, KeyError, TypeError):
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            resolved = [x for x in pool.map(lookup, pending) if x is not None]
        with _ledger_lock(self.ledger):
            current = json.loads(self.ledger.read_text())
            count = 0
            for key, generation, cost in resolved:
                row = current.get(key, {})
                if row.get('cost_usd') is None and row.get('generation_id') == generation:
                    row.update(cost_usd=cost, cost_source='generation.total_cost',
                               cost_reconciled_at=time.time())
                    count += 1
            if count:
                _atomic_json(self.ledger, current)
        return dict(queried=len(pending), reconciled=count, unresolved=len(pending)-count)

    def _record_batch(self, batch: dict) -> None:
        usage = batch.get('usage') or {}
        cost = usage.get('cost')
        self._account(batch['id'], status=batch.get('status'), usage=usage,
                      cost_usd=float(cost) if cost is not None else None,
                      cost_source='batch.usage.cost' if cost is not None else 'unreported')
        print(json.dumps({'api_batch': batch['id'], 'status': batch.get('status'),
                          'cost_usd': cost, 'usage': usage}), flush=True)

    @staticmethod
    def _normalize_sync_usage(result: dict, body: dict) -> bool:
        """Bound successful synchronous replies whose cached usage is zero/missing.

        OpenRouter's exact-response cache can return the original generation with
        an empty ``usage`` object (or zero token counts), including when the first
        HTTP attempt is itself a cache hit.  Such a reply is still a full model
        action for the experiment, so charge its entire requested completion
        allowance rather than allowing it to bypass the execution budget.

        Returns whether the response was changed.  Provider-reported dollar cost
        is deliberately left intact: a response-cache hit may genuinely be free.
        """
        usage = result.get('usage')
        if not isinstance(usage, dict):
            usage = {}
            result['usage'] = usage
        try:
            completion = int(usage.get('completion_tokens', 0) or 0)
        except (TypeError, ValueError):
            return False  # Budget validation will reject malformed provider data.
        if completion and usage.get('prompt_tokens') is not None:
            return False
        choices = result.get('choices')
        if not isinstance(choices, list) or not choices:
            return False
        if not completion:
            usage['completion_tokens'] = int(body['max_tokens'])
        if usage.get('prompt_tokens') is None:
            usage['prompt_tokens'] = 0
            result['prompt_tokens_unreported'] = True
        retries = int(result.get('_transport_retry_count', 0) or 0)
        if retries:
            result['usage_conservative_after_transport_retry'] = True
        else:
            result['usage_conservative_after_response_cache'] = True
        return True

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "solo2social",
        }

    @staticmethod
    def _read_json_response(response, deadline):
        # OpenRouter may send whitespace while waiting for a provider. A socket
        # inactivity timeout alone lets those keepalives extend a call forever.
        if not hasattr(response, 'read1'):
            return json.load(response)  # in-memory test/file responses
        chunks = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('OpenRouter response exceeded elapsed-time deadline')
            sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
            if sock is not None:
                sock.settimeout(remaining)
            chunk = response.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return json.loads(b''.join(chunks))

    def _json(self, method: str, url: str, body: dict | None = None) -> dict:
        retries = max(0, int(self.config.get('transport_retries', 4)))
        retryable_http = {408, 429, 500, 502, 503, 504}
        headers = self.headers
        if method == 'POST' and url.endswith('/v1/chat/completions'):
            # Exact retries can otherwise buy two stochastic generations when
            # inference succeeds but the response connection is lost.
            headers |= {'X-OpenRouter-Cache': 'true',
                        'X-OpenRouter-Cache-TTL': str(self.config.get('response_cache_ttl', 3600))}
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                url, method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers=headers)
            try:
                deadline = time.monotonic() + self.timeout
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = self._read_json_response(response, deadline)
                    if attempt and method == 'POST' and url.endswith('/v1/chat/completions'):
                        result['_transport_retry_count'] = attempt
                    return result
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode(errors='replace')
                except Exception:
                    detail = str(exc)
                if exc.code not in retryable_http or attempt >= retries:
                    raise RuntimeError(
                        f"OpenRouter request failed ({exc.code}): {detail[:1000]}") from exc
                delay = min(30, max(1, int(exc.headers.get('Retry-After', 0) or 0), 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.IncompleteRead, json.JSONDecodeError) as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"OpenRouter transport failed after {attempt + 1} attempts: {exc}") from exc
                delay = min(30, 2 ** attempt)
            print(json.dumps({'openrouter_transport_retry': attempt + 1,
                              'max_retries': retries, 'delay_seconds': delay,
                              'method': method}), flush=True)
            time.sleep(delay)
        raise AssertionError('unreachable')

    @staticmethod
    def _batch_result(item: dict) -> dict:
        if item.get("error"):
            error = item['error']
            if (isinstance(error, dict) and error.get('type') == 'upstream_error'
                    and 'MALFORMED_FUNCTION_CALL' in error.get('message', '')):
                # A rejected model generation is an experimental outcome. Preserve it
                # in the response cache so resumption cannot buy a free new attempt.
                return {'generation_error': 'MALFORMED_FUNCTION_CALL',
                        'usage_unavailable': True,
                        'choices': [{'message': {'role': 'assistant', 'content': ''}}]}
            raise RuntimeError(f"OpenRouter batch item failed: {str(item['error'])[:1000]}")
        response = item.get("response", item.get("body", item))
        if isinstance(response, dict) and "body" in response:
            if response.get("status_code", 200) >= 400:
                raise RuntimeError(
                    f"OpenRouter batch item failed ({response['status_code']}): "
                    f"{str(response['body'])[:1000]}"
                )
            response = response["body"]
        if not isinstance(response, dict) or "choices" not in response:
            raise RuntimeError(f"Malformed OpenRouter batch result: {str(item)[:1000]}")
        return response

    def _batch_completions(self, bodies: list[dict]) -> list[dict]:
        requests = [
            {"custom_id": f"request-{index}", "body": body | {"model": self.model}}
            for index, body in enumerate(bodies)
        ]
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.model,
            "requests": requests,
        }
        key = _digest(payload)
        cache = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        record = json.loads(cache.read_text()) if cache is not None and cache.exists() else {}
        batch_id = record.get("id")
        if not batch_id:
            reservation = 'pending-' + _digest([self.run_id, key])
            prior = self._account(reservation)
            if prior.get('batch_id'):
                batch_id = prior['batch_id']
            elif prior.get('status') == 'submitting':
                raise RuntimeError('Ambiguous prior batch submission; reconcile ledger before resubmitting')
        if not batch_id:
            self._reserve(reservation, bodies)
            created = self._json("POST", self.api_root + "/beta/batches", payload)
            batch_id = created.get("id")
            if not batch_id:
                raise RuntimeError(f"OpenRouter did not return a batch ID: {str(created)[:1000]}")
            if cache is not None:
                _atomic_json(cache, {"id": batch_id, "status": created.get("status")})
            self._account(batch_id, reserved_usd=self._account(reservation)['reserved_usd'],
                          status=created.get('status'), request_count=len(bodies), phase=self.phase)
            self._account(reservation, batch_id=batch_id, status='transferred', reserved_usd=0, cost_usd=0)

        poll_failures = 0
        while True:
            try:
                batch = self._json("GET", self.api_root + f"/beta/batches/{batch_id}")
                poll_failures = 0
            except (RuntimeError, urllib.error.URLError) as exc:
                cause = exc.__cause__ if isinstance(exc, RuntimeError) else exc
                transient = (isinstance(cause, urllib.error.HTTPError)
                             and cause.code in {404, 408, 429, 500, 502, 503, 504})
                transient |= isinstance(cause, urllib.error.URLError) and not isinstance(cause, urllib.error.HTTPError)
                if not transient or poll_failures >= 60:
                    raise
                poll_failures += 1
                time.sleep(self.poll_seconds)
                continue
            status = batch.get("status")
            if cache is not None:
                _atomic_json(cache, {"id": batch_id, "status": status})
            if status in TERMINAL_BATCH_STATES:
                break
            time.sleep(self.poll_seconds)
        if status != "completed":
            self._record_batch(batch | {'id': batch_id})
            raise RuntimeError(
                f"OpenRouter batch {batch_id} ended in {status}: "
                f"{str(batch.get('errors') or batch.get('error') or '')[:1000]}"
            )
        items = batch.get("results")
        self._record_batch(batch | {'id': batch_id})
        if not isinstance(items, list) or len(items) != len(bodies):
            raise RuntimeError(f"OpenRouter batch {batch_id} returned the wrong result count")
        by_id = {item.get("custom_id"): self._batch_result(item) for item in items}
        results = [by_id.get(f"request-{index}") for index in range(len(bodies))]
        if any(result is None for result in results):
            raise RuntimeError(f"OpenRouter batch {batch_id} omitted a custom request ID")
        batch_usage = batch.get('usage') or {}
        for result in results:
            result['batch_id'] = batch_id
            result['batch_usage'] = batch_usage
            if len(results) == 1 and batch_usage.get('cost') is not None:
                result.setdefault('usage', {})['cost'] = batch_usage['cost']
                result['cost_source'] = 'batch.usage.cost'
        if cache is not None:
            # Keep only the response required to resume an interrupted benchmark round.
            _atomic_json(cache, {"id": batch_id, "status": status, "results": results})
        return results

    def _check_provider_response(self, result, key, cache):
        error = result.get('error')
        if not error:
            return
        if not isinstance(error, dict):
            error = {'message': str(error)}
        usage = result.get('usage') or {}
        # Never refund an uncertain generation. Each new attempt reserves its
        # own allowance; the old reservation remains until cost is reported.
        self._account(key, status='provider_error', generation_id=result.get('id'),
                      usage=usage, cost_usd=usage.get('cost'),
                      provider_error=error,
                      cost_source='provider' if usage.get('cost') is not None else 'unreported')
        if cache is not None:
            _atomic_json(cache, {'status': 'provider_error', 'ledger_key': key,
                                 'result': result})
        raise ProviderResponseError(error.get('code'), str(error.get('message', ''))[:500])

    def chat_completions(self, bodies: list[dict]) -> list[dict]:
        if self.batch:
            return self._chat_completions_once(bodies)
        results = []
        for body in bodies:
            retries = max(0, int(self.config.get('provider_response_retries', 4)))
            for attempt in range(retries + 1):
                try:
                    results.extend(self._chat_completions_once([body]))
                    break
                except ProviderResponseError as exc:
                    if not exc.retryable or attempt == retries:
                        raise
                    print(json.dumps({'openrouter_provider_response_retry': attempt + 1,
                                      'max_retries': retries, 'error': str(exc)}), flush=True)
                    time.sleep(min(30, 2 ** attempt))
        return results

    def _chat_completions_once(self, bodies: list[dict]) -> list[dict]:
        if not bodies:
            return []
        if self.batch:
            payload = {
                "endpoint": "/v1/chat/completions",
                "model": self.model,
                "requests": [
                    {"custom_id": f"request-{index}", "body": body | {"model": self.model}}
                    for index, body in enumerate(bodies)
                ],
            }
            cache = self.cache_dir / f"{_digest(payload)}.json" if self.cache_dir is not None else None
            if cache is not None and cache.exists():
                saved = json.loads(cache.read_text())
                if saved.get("status") == "completed" and len(saved.get("results", [])) == len(bodies):
                    return saved["results"]
            return self._batch_completions(bodies)
        results = []
        for body in bodies:
            cache_key = _digest(body | {'model': self.model})
            cache = self.cache_dir / (cache_key + '.json') if self.cache_dir else None
            if cache is not None and cache.exists():
                saved = json.loads(cache.read_text())
                if saved.get('status') == 'completed':
                    result = saved['result']
                    self._check_provider_response(result, saved.get('ledger_key', 'legacy-' + cache_key), cache)
                    changed = self._normalize_sync_usage(result, body)
                    if changed:
                        # Persist the repair so every later resume observes the
                        # same budget charge. Do not create or alter dollar cost.
                        _atomic_json(cache, saved | {'result': result})
                    if saved.get('ledger_key'):
                        usage = result.get('usage', {})
                        extra = {'usage_accounting': 'conservative_response_cache'} if changed else {}
                        self._account(saved['ledger_key'], status='completed',
                                      generation_id=result.get('id'), usage=usage,
                                      cost_usd=usage.get('cost'),
                                      cost_source='provider' if usage.get('cost') is not None else 'unreported', **extra)
                    results.append(result)
                    continue
            key = 'sync-' + uuid.uuid4().hex
            if cache is not None:
                saved = json.loads(cache.read_text()) if cache.exists() else {}
                if saved.get('status') in {'submitting', 'ambiguous'}:
                    prior_key = saved.get('ledger_key')
                    if prior_key:
                        self._account(prior_key, status='ambiguous',
                                      transport_retry_started_at=time.time())
            self._reserve(key, [body])
            if cache is not None:
                _atomic_json(cache, {'status': 'submitting', 'ledger_key': key})
            try:
                result = self._json('POST', self.api_root + '/v1/chat/completions', body | {'model': self.model})
            except RuntimeError as exc:
                cause = exc.__cause__
                if isinstance(cause, urllib.error.HTTPError) and cause.code in {400, 401, 402, 403, 404, 422, 429}:
                    self._account(key, status='rejected', cost_usd=0., reserved_usd=0.,
                                  http_status=cause.code, cost_source='request_rejected_before_inference')
                    if cache is not None:
                        _atomic_json(cache, {'status': 'rejected', 'ledger_key': key, 'http_status': cause.code})
                else:
                    self._account(key, status='ambiguous', transport_error=str(exc)[:500])
                    if cache is not None:
                        _atomic_json(cache, {'status': 'ambiguous', 'ledger_key': key,
                                             'transport_error': str(exc)[:500]})
                raise
            self._check_provider_response(result, key, cache)
            self._normalize_sync_usage(result, body)
            usage = result.setdefault('usage', {})
            cost = usage.get('cost')
            if cost is None and result.get('id'):
                try:
                    metadata = self._json('GET', self.api_root + '/v1/generation?id=' + result['id'])['data']
                    cost = metadata.get('total_cost')
                    if cost is not None:
                        result.setdefault('usage', {})['cost'] = cost
                except (RuntimeError, KeyError, TypeError):
                    pass  # Keep the conservative reservation until reconciliation.
            if cache is not None:
                _atomic_json(cache, {'status': 'completed', 'ledger_key': key, 'result': result})
            self._account(key, status='completed', generation_id=result.get('id'),
                          usage=usage, cost_usd=cost, cost_source='provider' if cost is not None else 'unreported')
            results.append(result)
        return results

    def clear_cache(self) -> None:
        """Drop consumed responses after the enclosing round is checkpointed."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        for path in self.cache_dir.glob("*.json"):
            path.unlink()


def usage_report(path: Path) -> dict:
    data = json.loads(path.read_text()) if path.exists() else {}
    runs = {}
    for row in data.values():
        if row.get('status') == 'transferred':
            continue
        out = runs.setdefault(row['run'], dict(reported_cost_usd=0., unresolved_reserved_usd=0.,
                            requests=0, unreported_cost_records=0, byok=False))
        out['requests'] += row.get('request_count', 1)
        if row.get('cost_usd') is None:
            out['unreported_cost_records'] += 1
            out['unresolved_reserved_usd'] += row.get('reserved_usd', 0)
        else:
            out['reported_cost_usd'] += row['cost_usd']
        out['byok'] |= bool(row.get('usage', {}).get('is_byok', False))
    return {'runs': runs, 'reported_cost_usd': sum(x['reported_cost_usd'] for x in runs.values()),
            'unresolved_reserved_usd': sum(x['unresolved_reserved_usd'] for x in runs.values()),
            'note': 'Provider-reported OpenRouter charges; BYOK excludes provider bills. Legacy requests without retained IDs are not recoverable here.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Report/reconcile OpenRouter spending; run through Slurm.')
    parser.add_argument('--audit-cache', type=Path, help='Recover batch billing for retained per-run caches')
    parser.add_argument('--reconcile-generations', action='store_true',
                        help='Resolve missing charges from saved generation IDs; no inference calls')
    parser.add_argument('--ledger', type=Path, default=Path(__file__).resolve().parents[1] / 'runs/openrouter_usage/ledger.json')
    args = parser.parse_args()
    if args.reconcile_generations:
        client = OpenRouterClient({'name': 'billing-audit', 'usage_ledger': str(args.ledger),
                                   'timeout': 30}, args.ledger.parent / 'api_cache')
        print(json.dumps(client.reconcile_generation_costs()), flush=True)
    if args.audit_cache:
        for config_path in sorted(args.audit_cache.glob('*/seed_*/config.json')):
            config = json.loads(config_path.read_text())['model'] | {'usage_ledger': str(args.ledger)}
            for directory in ('api_batches', 'api_batches_replay'):
                client = OpenRouterClient(config, config_path.parent / directory)
                for cache in sorted((config_path.parent / directory).glob('*.json')):
                    record = json.loads(cache.read_text())
                    if record.get('id'):
                        batch = client._json('GET', client.api_root + '/beta/batches/' + record['id'])
                        client._record_batch(batch | {'id': record['id']})
    print(json.dumps(usage_report(args.ledger), indent=2))
