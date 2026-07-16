import builtins

import pytest

from src.llm_assistant import (
    COST_PER_1K_TOKENS,
    PROVIDER_MODELS,
    PROVIDERS,
    UsageTracker,
    estimate_cost,
    estimate_tokens,
    load_api_key,
    stream_openai,
    stream_llm,
    stream_with_fallback,
)


def test_estimate_tokens_nonzero():
    assert estimate_tokens('hello world this is a test') > 0


def test_estimate_tokens_empty_string():
    assert estimate_tokens('') == 0


def test_estimate_cost_zero_for_unknown_model():
    cost = estimate_cost('hello', 'world', model='unknown-model')

    assert cost == 0.0


def test_estimate_cost_positive_for_known_model():
    cost = estimate_cost('hello ' * 100, 'world ' * 100, model='gpt-4o-mini')

    assert cost > 0


def test_provider_constants_include_expected_options():
    assert PROVIDERS == ['ollama', 'openai', 'anthropic', 'none']


def test_provider_models_include_hosted_options():
    assert PROVIDER_MODELS['openai']
    assert PROVIDER_MODELS['anthropic']


def test_cost_table_contains_multiple_models():
    assert len(COST_PER_1K_TOKENS) >= 3


def test_load_api_key_returns_none_when_missing(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    assert load_api_key('openai') is None


def test_load_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    assert load_api_key('openai') == 'sk-test'


def test_load_api_key_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath('.env').write_text('OPENAI_API_KEY=sk-dotenv\n')

    assert load_api_key('openai') == 'sk-dotenv'


def test_usage_tracker_records():
    tracker = UsageTracker()
    tracker.record('openai', 'gpt-4o-mini', 'hello', 'world')

    assert tracker.total_tokens() > 0
    assert len(tracker.to_dataframe()) == 1


def test_usage_tracker_starts_empty():
    tracker = UsageTracker()

    assert tracker.total_cost() == 0.0
    assert tracker.total_tokens() == 0
    assert tracker.to_dataframe().empty


def test_usage_tracker_summary_format():
    tracker = UsageTracker()
    tracker.record(
        'anthropic',
        'claude-haiku-4-5-20251001',
        'a ' * 50,
        'b ' * 50,
    )
    summary = tracker.summary()

    assert 'call' in summary and 'token' in summary and '$' in summary


def test_stream_with_fallback_none_provider():
    gen, source = stream_with_fallback('test', 'none', None, 'fallback')
    result = ''.join(gen)

    assert result == 'fallback'
    assert source == 'deterministic'


def test_stream_llm_none_provider_yields_nothing():
    assert list(stream_llm('test prompt', 'none', None)) == []


def test_stream_llm_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match='Unknown provider'):
        list(stream_llm('test prompt', 'not-a-provider', None))


def test_stream_openai_import_error_when_package_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'openai':
            raise ImportError('mocked missing openai')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    gen = stream_openai('test prompt', api_key='sk-test')

    with pytest.raises(ImportError, match='pip install openai'):
        next(gen)


def test_stream_with_fallback_missing_hosted_key_uses_deterministic(monkeypatch):
    import src.llm_assistant as llm_assistant

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(
        llm_assistant,
        'stream_ollama',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('offline')),
    )

    gen, source = stream_with_fallback(
        'test',
        'openai',
        'gpt-4o-mini',
        'fallback',
    )
    result = ''.join(gen)

    assert result == 'fallback'
    assert source == 'deterministic'
