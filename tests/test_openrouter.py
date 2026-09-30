"""OpenRouter transport and provider dispatch without network access."""

import json
import pickle
import pytest
import urllib.error

from agent import OpenRouterDecisionEngine, VLLMDecisionEngine
from openrouter import OpenRouterClient, ProviderResponseError, SpendingLimitError, _digest


@pytest.mark.parametrize('error', [SpendingLimitError('cap reached', False),
                                  SpendingLimitError('in flight', True),
                                  ProviderResponseError(504, 'timeout'),
                                  ProviderResponseError(401, 'unauthorized')])
def test_transport_errors_survive_worker_serialization(error):
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.__dict__ == error.__dict__


@pytest.mark.parametrize('cached', [False, True])
def test_provider_error_retries_without_replaying_poisoned_cache(monkeypatch, tmp_path, cached):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    monkeypatch.setattr('openrouter.time.sleep', lambda _: None)
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    body = {'messages': [], 'max_tokens': 13}
    error = {'id': 'failed-generation', 'error': {'code': 504, 'message': 'Provider timed out'}, 'usage': {}}
    cache = client.cache_dir / (_digest(body | {'model': client.model}) + '.json')
    if cached:
        client._reserve('original-request', [body])
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({'status': 'completed', 'ledger_key': 'original-request', 'result': error}))
    calls = []
    def request(*args):
        calls.append(args)
        if not cached and len(calls) == 1:
            return error
        return {'id': 'valid-generation', 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'cost': .001},
                'choices': [{'message': {'content': 'valid answer'}}]}
    monkeypatch.setattr(client, '_json', request)
    result = client.chat_completions([body])[0]
    assert result['choices'][0]['message']['content'] == 'valid answer'
    assert len(calls) == (1 if cached else 2)
    rows = list(json.loads(client.ledger.read_text()).values())
    failed = [r for r in rows if r['status'] == 'provider_error']
    assert len(failed) == 1 and failed[0]['reserved_usd'] > 0 and failed[0].get('cost_usd') is None
    client.chat_completions([body])
    assert len(calls) == (1 if cached else 2)  # successful answer still reused


def test_provider_errors_have_bounded_retries_and_preserve_cap(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    monkeypatch.setattr('openrouter.time.sleep', lambda _: None)
    client = OpenRouterClient({'name': 'test/model', 'provider_response_retries': 2}, tmp_path / 'run/cache')
    monkeypatch.setattr(client, '_json', lambda *args: {'error': {'code': 503, 'message': 'busy'}})
    body = {'messages': [], 'max_tokens': 13}
    with pytest.raises(ProviderResponseError):
        client.chat_completions([body])
    rows = list(json.loads(client.ledger.read_text()).values())
    assert len(rows) == 3 and all(r['status'] == 'provider_error' and r['reserved_usd'] > 0 for r in rows)


def test_nonretryable_provider_response_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    monkeypatch.setattr(client, '_json', lambda *args: {'error': {'code': 401, 'message': 'unauthorized'}})
    with pytest.raises(ProviderResponseError):
        client.chat_completions([{'messages': [], 'max_tokens': 13}])
    assert len(json.loads(client.ledger.read_text())) == 1


def test_provider_retry_cannot_spend_past_existing_limit(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    monkeypatch.setattr('openrouter.time.sleep', lambda _: None)
    client = OpenRouterClient({'name': 'test/model', 'total_cost_limit_usd': .006,
                               'timeout': 0}, tmp_path / 'run/cache')
    monkeypatch.setattr(client, '_json', lambda *args: {'error': {'code': 504, 'message': 'timeout'}})
    with pytest.raises(SpendingLimitError):
        client.chat_completions([{'messages': [], 'max_tokens': 13}])
    rows = list(json.loads(client.ledger.read_text()).values())
    assert len(rows) == 1 and rows[0]['reserved_usd'] <= .006


def test_missing_prompt_usage_keeps_reported_completion():
    result = {'usage': {'completion_tokens': 5, 'cost': .001},
              'choices': [{'message': {'content': 'answer'}}]}
    assert OpenRouterClient._normalize_sync_usage(result, {'max_tokens': 13})
    assert result['usage'] == {'completion_tokens': 5, 'prompt_tokens': 0, 'cost': .001}
    assert result['prompt_tokens_unreported']


def test_generation_reconciliation_preserves_unverified_reservations(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'cache')
    records = {
        'valid': dict(generation_id='ok', cost_usd=None, reserved_usd=.1),
        'unknown': dict(cost_usd=None, reserved_usd=.1),
        'mismatch': dict(generation_id='wrong', cost_usd=None, reserved_usd=.1),
        'settled': dict(generation_id='paid', cost_usd=.02, reserved_usd=.1),
        'missing': dict(generation_id='missing', cost_usd=None, reserved_usd=.1),
    }
    client.ledger.write_text(json.dumps(records))
    def fetch(method, url):
        if url.endswith('id=missing'):
            raise RuntimeError('not found')
        return {'data': {'id': 'ok', 'total_cost': .003}}
    monkeypatch.setattr(client, '_json', fetch)
    result = client.reconcile_generation_costs()
    after = json.loads(client.ledger.read_text())
    assert result == dict(queried=3, reconciled=1, unresolved=2)
    assert after['valid']['cost_usd'] == .003
    assert after['valid']['reserved_usd'] == .1
    for key in ('unknown', 'mismatch', 'settled', 'missing'):
        assert after[key] == records[key]


@pytest.fixture(autouse=True)
def isolated_accounting(monkeypatch, tmp_path):
    original = OpenRouterClient.__init__
    def initialize(self, config, cache_dir=None):
        original(self, {'pricing': {'prompt': '0.000001', 'completion': '0.000002'},
                        'usage_ledger': str(tmp_path / 'ledger.json')} | config, cache_dir or tmp_path / 'cache')
    monkeypatch.setattr(OpenRouterClient, '__init__', initialize)


def test_resumable_batch_and_result_order(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")
    calls = []

    class Response:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.value).encode()

    def urlopen(request, timeout):
        calls.append((request.method, request.full_url))
        if request.method == "POST":
            body = json.loads(request.data)
            assert body["model"] == "google/gemini-3.8-flash"
            assert all(x["body"]["model"] == body["model"] for x in body["requests"])
            assert request.headers["Authorization"] == "Bearer test-key"
            return Response({"id": "batch-1", "status": "validating"})
        result = lambda text: {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3}}
        return Response({"id": "batch-1", "status": "completed", "results": [
            {"custom_id": "request-1", "response": {"status_code": 200, "body": result("b")}},
            {"custom_id": "request-0", "response": {"status_code": 200, "body": result("a")}},
        ]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OpenRouterClient({"name": "google/gemini-3.8-flash:batch",
        "batch_poll_seconds": 2}, tmp_path)
    bodies = [{"messages": [], "max_tokens": 8}, {"messages": [], "max_tokens": 8}]
    assert [x["choices"][0]["message"]["content"] for x in client.chat_completions(bodies)] == ["a", "b"]
    assert len(calls) == 2
    client.chat_completions(bodies)
    assert len(calls) == 2


def test_batch_poll_retries_same_id_after_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model:batch'}, tmp_path)
    calls = []
    def request(method, url, body=None):
        calls.append((method, url))
        if method == 'POST':
            return {'id': 'batch-existing', 'status': 'validating'}
        if len(calls) == 2:
            try:
                raise urllib.error.HTTPError(url, 404, 'Not found', {}, None)
            except urllib.error.HTTPError as exc:
                raise RuntimeError('OpenRouter request failed (404)') from exc
        return {'status': 'completed', 'results': [{'custom_id': 'request-0',
            'response': {'body': {'choices': [{'message': {'content': 'done'}}]}}}]}
    monkeypatch.setattr(client, '_json', request)
    monkeypatch.setattr('openrouter.time.sleep', lambda _: None)
    assert client.chat_completions([{'messages': [], 'max_tokens': 8}])[0]['choices'][0]['message']['content'] == 'done'
    assert sum(method == 'POST' for method, _ in calls) == 1
    assert calls[1] == calls[2]


def test_vllm_facade_dispatches_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")
    engine = VLLMDecisionEngine({"provider": "openrouter", "name": "test/model"})
    assert isinstance(engine, OpenRouterDecisionEngine)


def test_malformed_generation_is_cached_without_resubmission(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model:batch'}, tmp_path)
    calls = []
    def request(method, url, body=None):
        calls.append(method)
        if method == 'POST':
            return {'id': 'batch-failed-generation', 'status': 'validating'}
        return {'status': 'completed', 'results': [{'custom_id': 'request-0',
            'error': {'type': 'upstream_error',
                      'message': 'Gemini blocked the response: MALFORMED_FUNCTION_CALL'}}]}
    monkeypatch.setattr(client, '_json', request)
    bodies = [{'messages': [], 'max_tokens': 20}]
    result = client.chat_completions(bodies)
    assert result[0]['generation_error'] == 'MALFORMED_FUNCTION_CALL'
    assert 'usage' not in result[0]
    assert client.chat_completions(bodies) == result
    assert calls == ['POST', 'GET']
    with pytest.raises(RuntimeError, match='batch item failed'):
        client._batch_result({'error': {'type': 'upstream_error', 'message': 'Payment required'}})


def test_batch_cost_survives_cache_cleanup_and_is_not_counted_twice(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model:batch'}, tmp_path / 'run/cache')
    def request(method, url, body=None):
        if method == 'POST':
            return {'id': 'costed-batch', 'status': 'validating'}
        return {'status': 'completed', 'usage': {'cost': .0123, 'prompt_tokens': 2, 'completion_tokens': 3},
                'results': [{'custom_id': 'request-0', 'response': {'body': {
                    'choices': [{'message': {'content': 'done'}}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}}}]}
    monkeypatch.setattr(client, '_json', request)
    body = [{'messages': [], 'max_tokens': 8}]
    assert client.chat_completions(body)[0]['usage']['cost'] == .0123
    client.chat_completions(body)
    client.clear_cache()
    from openrouter import usage_report
    assert usage_report(client.ledger)['reported_cost_usd'] == .0123
    assert not list(client.cache_dir.glob('*.json'))


def test_shared_cap_includes_pending_requests(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    a = OpenRouterClient({'name': 'test/model:batch', 'total_cost_limit_usd': .006}, tmp_path / 'a/cache')
    b = OpenRouterClient({'name': 'test/model:batch', 'total_cost_limit_usd': .006,
                         'timeout': 0}, tmp_path / 'b/cache')
    a._reserve('request-a', [{'messages': [], 'max_tokens': 8}])
    with pytest.raises(RuntimeError, match='shared spending limit'):
        b._reserve('request-b', [{'messages': [], 'max_tokens': 8}])


def test_ambiguous_sync_submission_is_conservatively_reserved_before_retry(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    calls = []
    def fail(method, url, body=None):
        calls.append(method)
        raise RuntimeError('connection lost after submission')
    monkeypatch.setattr(client, '_json', fail)
    body = [{'messages': [], 'max_tokens': 8}]
    with pytest.raises(RuntimeError, match='connection lost'):
        client.chat_completions(body)
    with pytest.raises(RuntimeError, match='connection lost'):
        client.chat_completions(body)
    assert calls == ['POST', 'POST']
    ledger = json.loads(client.ledger.read_text())
    assert len(ledger) == 2 and all(row['status'] == 'ambiguous' for row in ledger.values())
    assert all(row['reserved_usd'] > 0 for row in ledger.values())


def test_sync_transport_retries_use_exact_response_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model', 'transport_retries': 2}, tmp_path / 'run/cache')
    calls, sleeps = [], []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'id': 'generation-test',
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0},
                'choices': [{'message': {'content': 'done'}}]}).encode()
    def request(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise TimeoutError('read timed out')
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', request)
    monkeypatch.setattr('openrouter.time.sleep', sleeps.append)
    result = client.chat_completions([{'messages': [], 'max_tokens': 8}])
    assert result[0]['choices'][0]['message']['content'] == 'done'
    assert sleeps == [1, 2] and len(calls) == 3
    assert all(req.headers['X-openrouter-cache'] == 'true' for req in calls)
    assert result[0]['usage']['completion_tokens'] == 8
    assert result[0]['usage_conservative_after_transport_retry']


@pytest.mark.parametrize('reported_usage', [{}, {
    'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0,
}])
def test_first_attempt_response_cache_is_conservatively_charged(
        monkeypatch, tmp_path, reported_usage):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    monkeypatch.setattr(client, '_json', lambda method, url, body=None: {
        'id': 'cached-generation', 'usage': dict(reported_usage),
        'choices': [{'message': {'content': 'cached answer'}}],
    })
    result = client.chat_completions([{'messages': [], 'max_tokens': 13}])[0]
    assert result['usage']['completion_tokens'] == 13
    assert result['usage']['prompt_tokens'] == 0
    assert result['usage_conservative_after_response_cache']
    if 'cost' in reported_usage:
        assert result['usage']['cost'] == 0


def test_saved_zero_usage_response_is_repaired_without_resubmission(
        monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    cache_dir = tmp_path / 'run/cache'
    client = OpenRouterClient({'name': 'test/model'}, cache_dir)
    body = {'messages': [], 'max_tokens': 11}
    cache_dir.mkdir(parents=True)
    cache = cache_dir / (_digest(body | {'model': client.model}) + '.json')
    cache.write_text(json.dumps({
        'status': 'completed', 'ledger_key': 'existing-request',
        'result': {'id': 'cached-generation', 'usage': {},
                   'choices': [{'message': {'content': 'cached answer'}}]},
    }))
    monkeypatch.setattr(client, '_json', lambda *args, **kwargs:
                        pytest.fail('A saved response must not be submitted again'))
    result = client.chat_completions([body])[0]
    assert result['usage'] == {'completion_tokens': 11, 'prompt_tokens': 0}
    assert result['usage_conservative_after_response_cache']
    assert json.loads(cache.read_text())['result']['usage']['completion_tokens'] == 11
    ledger = json.loads(client.ledger.read_text())
    assert ledger['existing-request']['usage_accounting'] == 'conservative_response_cache'


def test_nonzero_sync_usage_is_not_rewritten(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    usage = {'prompt_tokens': 5, 'completion_tokens': 3, 'cost': .001}
    monkeypatch.setattr(client, '_json', lambda method, url, body=None: {
        'id': 'fresh-generation', 'usage': dict(usage),
        'choices': [{'message': {'content': 'fresh answer'}}],
    })
    result = client.chat_completions([{'messages': [], 'max_tokens': 13}])[0]
    assert result['usage'] == usage
    assert 'usage_conservative_after_response_cache' not in result


def test_sync_response_resumes_without_another_charge(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    calls = []
    def request(method, url, body=None):
        calls.append(method)
        return {'id': 'generation-test', 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'cost': .001},
                'choices': [{'message': {'content': 'done'}}]}
    monkeypatch.setattr(client, '_json', request)
    body = [{'messages': [], 'max_tokens': 8}]
    first = client.chat_completions(body)
    assert client.chat_completions(body) == first
    assert calls == ['POST']
    from openrouter import usage_report
    assert usage_report(client.ledger)['reported_cost_usd'] == .001


def test_completed_response_survives_accounting_failure(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model'}, tmp_path / 'run/cache')
    calls = []
    def request(method, url, body=None):
        calls.append(method)
        return {'id': 'paid-generation', 'usage': {'completion_tokens': 3, 'cost': .001},
                'choices': [{'message': {'content': 'paid answer'}}]}
    monkeypatch.setattr(client, '_json', request)
    account = client._account
    def failing(key, **updates):
        if updates.get('status') == 'completed':
            raise RuntimeError('Ledger lock timed out')
        return account(key, **updates)
    monkeypatch.setattr(client, '_account', failing)
    with pytest.raises(RuntimeError, match='Ledger lock'):
        client.chat_completions([{'messages': [], 'max_tokens': 8}])
    monkeypatch.setattr(client, '_account', account)
    answer = client.chat_completions([{'messages': [], 'max_tokens': 8}])[0]
    assert answer['choices'][0]['message']['content'] == 'paid answer'
    assert calls == ['POST']
    from openrouter import usage_report
    assert usage_report(client.ledger)['reported_cost_usd'] == .001


def test_admission_waits_for_inflight_cost_to_settle(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    client = OpenRouterClient({'name': 'test/model', 'total_cost_limit_usd': .006}, tmp_path / 'run/cache')
    client._reserve('first', [{'messages': [], 'max_tokens': 8}])
    waited = []
    def settle(seconds):
        waited.append(seconds)
        client._account('first', status='completed', cost_usd=.001)
    monkeypatch.setattr('openrouter.time.sleep', settle)
    client._reserve('second', [{'messages': [], 'max_tokens': 8}])
    assert waited
    assert json.loads(client.ledger.read_text())['second']['status'] == 'submitting'


def test_local_threads_do_not_exhaust_filesystem_lock_timeout(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from openrouter import _ledger_lock
    path = tmp_path / 'ledger.json'
    def worker(i):
        with _ledger_lock(path, timeout=.001):
            time.sleep(.005)
        return i
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert list(pool.map(worker, range(32))) == list(range(32))


def test_stale_ledger_lock_fails_loudly(tmp_path):
    from openrouter import _ledger_lock
    path = tmp_path / 'ledger.json'
    path.with_suffix('.lockdir').mkdir()
    with pytest.raises(RuntimeError, match='Ledger lock timed out'):
        with _ledger_lock(path, timeout=0):
            pytest.fail('Acquired an existing lock')


def test_terminal_slurm_owner_lock_is_recovered(monkeypatch, tmp_path):
    from openrouter import _ledger_lock
    path = tmp_path / 'ledger.json'
    lock = path.with_suffix('.lockdir')
    lock.mkdir()
    (lock / 'owner.json').write_text(json.dumps({
        'host': 'different-node', 'pid': 123, 'job': '12345', 'created_at': 0,
    }))
    monkeypatch.setattr('openrouter._slurm_job_live', lambda job: False)
    with _ledger_lock(path, timeout=0, stale_after=0):
        assert lock.exists()
    assert not lock.exists()


def test_live_slurm_owner_lock_is_not_stolen(monkeypatch, tmp_path):
    from openrouter import _ledger_lock
    path = tmp_path / 'ledger.json'
    lock = path.with_suffix('.lockdir')
    lock.mkdir()
    (lock / 'owner.json').write_text(json.dumps({
        'host': 'different-node', 'pid': 123, 'job': '12345', 'created_at': 0,
    }))
    monkeypatch.setattr('openrouter._slurm_job_live', lambda job: True)
    with pytest.raises(RuntimeError, match='Ledger lock timed out'):
        with _ledger_lock(path, timeout=0, stale_after=0):
            pytest.fail('Stole a live lock')


@pytest.mark.parametrize('completion,prompt', [(9, 2), (0, 2), (-1, 2), (1, -1), (1.5, 2)])
def test_remote_usage_cannot_violate_budget(monkeypatch, completion, prompt):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    engine = OpenRouterDecisionEngine({'name': 'test/model'})
    monkeypatch.setattr(engine.client, 'chat_completions', lambda bodies: [{
        'usage': {'completion_tokens': completion, 'prompt_tokens': prompt},
        'choices': [{'message': {'content': 'FINAL: PULL 0'}}],
    }])
    with pytest.raises(RuntimeError, match='invalid usage'):
        engine.generate([{'system_prompt': 'test', 'state': 'test', 'seed': 0, 'max_tokens': 8}])


def test_requeued_job_can_recover_prior_allocation_lock(monkeypatch):
    import openrouter
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURM_RESTART_COUNT', '2')
    monkeypatch.setattr(openrouter, '_slurm_job_live', lambda job: True)
    old = dict(job='123', host='previous-node', pid=42, restart_count=1)
    assert openrouter._lock_owner_live(old) is False
    assert openrouter._lock_owner_live(dict(old, restart_count=2)) is True
    assert openrouter._lock_owner_live(dict(old, job='456')) is True
def test_response_keepalives_cannot_extend_elapsed_deadline(monkeypatch):
    from types import SimpleNamespace
    from openrouter import OpenRouterClient
    ticks = iter([0., 4., 11.])
    monkeypatch.setattr('openrouter.time.monotonic', lambda: next(ticks))
    timeouts = []
    response = SimpleNamespace(read1=lambda size: b' ',
        fp=SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=timeouts.append))))
    with pytest.raises(TimeoutError, match='elapsed-time'):
        OpenRouterClient._read_json_response(response, 10.)
    assert timeouts == [10., 6.]
    monkeypatch.setattr('openrouter.time.monotonic', lambda: 0.)
    chunks = iter([b' ', b'{"ok": true}', b''])
    response.read1 = lambda size: next(chunks)
    assert OpenRouterClient._read_json_response(response, 10.) == {'ok': True}
