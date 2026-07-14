"""Optional local LLM explanations for dataset assistant results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.automl import PreparedDataset


DEFAULT_OLLAMA_URL = 'http://localhost:11434/api/generate'
DEFAULT_OLLAMA_MODEL = 'llama3.1'

PROVIDERS = ['ollama', 'openai', 'anthropic', 'none']

PROVIDER_MODELS = {
    'ollama': ['llama3.1', 'llama3.2', 'mistral', 'phi3', 'gemma2'],
    'openai': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
    'anthropic': ['claude-haiku-4-5-20251001', 'claude-sonnet-4-6'],
}

COST_PER_1K_TOKENS = {
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    'gpt-4o': {'input': 0.005, 'output': 0.015},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'claude-haiku-4-5-20251001': {'input': 0.00025, 'output': 0.00125},
    'claude-sonnet-4-6': {'input': 0.003, 'output': 0.015},
}

DEFAULT_LLM_SYSTEM_PROMPT = (
    'You are a data science assistant. Answer concisely and ground claims '
    'in the provided statistics.'
)


def build_dataset_context(
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
) -> str:
    """Build a compact, factual context block for LLM explanations."""
    lines = [
        f"Rows: {profile.get('n_rows', 'unknown')}",
        f"Columns: {profile.get('n_columns', 'unknown')}",
        f"Missing cells: {profile.get('missing_cells', 'unknown')}",
        f"Duplicate rows: {profile.get('duplicate_rows', 'unknown')}",
    ]

    if prepared is not None:
        lines.extend([
            f"Task type: {prepared.task_type}",
            f"Target column: {prepared.target_col}",
            f"Feature count: {len(prepared.feature_cols)}",
            f"Numeric features: {', '.join(prepared.numeric_cols) or 'none'}",
            f"Categorical features: {', '.join(prepared.categorical_cols) or 'none'}",
        ])

    if results is not None and not results.empty:
        best = results.iloc[0]
        lines.append(f"Best baseline model: {best['model']}")
        lines.append(
            'Evaluated baseline models: '
            + ', '.join(results['model'].astype(str).tolist())
        )

        if prepared and prepared.task_type == 'classification':
            lines.extend([
                f"CV accuracy: {float(best['cv_accuracy']):.3f}",
                f"Test accuracy: {float(best['test_accuracy']):.3f}",
                f"Test macro F1: {float(best['test_f1_macro']):.3f}",
            ])
            score_rows = [
                f"{row.model}: test_accuracy={float(row.test_accuracy):.3f}, "
                f"test_f1_macro={float(row.test_f1_macro):.3f}"
                for row in results.head(8).itertuples()
            ]
        else:
            lines.extend([
                f"CV R2: {float(best['cv_r2']):.3f}",
                f"Test R2: {float(best['test_r2']):.3f}",
                f"Test MAE: {float(best['test_mae']):.3f}",
            ])
            score_rows = [
                f"{row.model}: test_r2={float(row.test_r2):.3f}, "
                f"test_mae={float(row.test_mae):.3f}"
                for row in results.head(8).itertuples()
            ]

        lines.append('Baseline score table: ' + '; '.join(score_rows))

    if feature_ranking is not None and not feature_ranking.empty:
        top_features = [
            f"{row.feature}={float(row.mutual_information):.4f}"
            for row in feature_ranking.head(8).itertuples()
        ]
        lines.append(f"Top mutual-information features: {', '.join(top_features)}")

    return '\n'.join(lines)


def build_dataset_prompt(context: str, question: str) -> str:
    """Create a grounded prompt that discourages unsupported claims."""
    return (
        'You are an ML dataset assistant. Answer naturally, but only use the '
        'computed facts below. It is allowed and expected to discuss model '
        'selection, feature importance, classification, regression, and evaluation '
        'metrics when those claims are grounded in the computed facts. If the facts '
        'are not enough, say what should be computed next. Do not suggest training '
        'models that are already listed as evaluated. You cannot execute code, '
        'train models, compute new metrics, access files, change dashboard state, '
        'or perform actions from this chat. If the user asks you to do something, '
        'do not refuse the ML topic; say you cannot run new computations from '
        'chat yet, summarize the existing computed results, and name the exact '
        'dashboard step or computation needed next. Do not say "I will compute", '
        '"I can perform", "I will run", or imply that new calculations were '
        'completed. Keep the answer concise and practical.\n\n'
        f'Computed facts:\n{context}\n\n'
        f'User question: {question}\n\n'
        'Answer:'
    )


def ask_ollama(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 30.0,
) -> str:
    """Ask a local Ollama model and return its text response."""
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
    }).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Local LLM request failed: {exc}') from exc

    answer = str(data.get('response', '')).strip()

    if not answer:
        raise RuntimeError('Local LLM returned an empty response.')

    return answer


def parse_ollama_stream_line(line: bytes) -> tuple[str, bool]:
    """Parse one Ollama streaming response line."""
    if not line:
        return '', False

    try:
        data = json.loads(line.decode('utf-8'))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Could not parse local LLM stream: {exc}') from exc

    return str(data.get('response', '')), bool(data.get('done', False))


def stream_ollama(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
):
    """Yield chunks from a local Ollama model as they arrive."""
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': True,
    }).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                chunk, done = parse_ollama_stream_line(line)

                if chunk:
                    yield chunk

                if done:
                    break
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f'Local LLM streaming failed: {exc}') from exc


def load_api_key(provider: str) -> str | None:
    """Load a hosted provider API key from env vars or Streamlit secrets."""
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
    }

    env_name = env_vars.get(provider)

    if env_name:
        key = os.environ.get(env_name)

        if key:
            return key

        key = _load_key_from_dotenv(env_name)

        if key:
            return key

    try:
        import streamlit as st

        key = st.secrets[provider]['api_key']

        if key:
            return str(key)
    except (ImportError, AttributeError, KeyError, FileNotFoundError, RuntimeError):
        return None

    return None


def _load_key_from_dotenv(env_name: str) -> str | None:
    """Read one key from a local .env file without requiring python-dotenv."""
    env_path = Path('.env')

    if not env_path.exists():
        return None

    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()

            if not line or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)

            if key.strip() == env_name:
                return value.strip().strip('"').strip("'") or None
    except OSError:
        return None

    return None


def estimate_tokens(text: str) -> int:
    """Estimate token count without adding tokenizer dependencies."""
    if not text:
        return 0

    return int(round(len(text.split()) * 1.3))


def estimate_cost(prompt: str, response: str, model: str | None) -> float:
    """Estimate hosted LLM cost from approximate token counts."""
    rates = COST_PER_1K_TOKENS.get(str(model), {'input': 0.0, 'output': 0.0})
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response)

    return (
        input_tokens * float(rates['input'])
        + output_tokens * float(rates['output'])
    ) / 1000.0


def stream_openai(
    prompt: str,
    model: str = 'gpt-4o-mini',
    api_key: str | None = None,
    max_tokens: int = 500,
    system: str = DEFAULT_LLM_SYSTEM_PROMPT,
) -> Generator[str, None, None]:
    """Yield streamed chunks from OpenAI's chat completion API."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError('openai package not installed. Run: pip install openai') from exc

    key = api_key or load_api_key('openai')

    if not key:
        raise ValueError(
            'No OpenAI API key found. Set OPENAI_API_KEY env var or add to st.secrets.'
        )

    client = openai.OpenAI(api_key=key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            yield delta


def stream_anthropic(
    prompt: str,
    model: str = 'claude-haiku-4-5-20251001',
    api_key: str | None = None,
    max_tokens: int = 500,
    system: str = DEFAULT_LLM_SYSTEM_PROMPT,
) -> Generator[str, None, None]:
    """Yield streamed chunks from Anthropic's messages API."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            'anthropic package not installed. Run: pip install anthropic'
        ) from exc

    key = api_key or load_api_key('anthropic')

    if not key:
        raise ValueError(
            'No Anthropic API key found. Set ANTHROPIC_API_KEY env var or '
            'add to st.secrets.'
        )

    client = anthropic.Anthropic(api_key=key)

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_llm(
    prompt: str,
    provider: str,
    model: str | None,
    api_key: str | None = None,
    max_tokens: int = 500,
) -> Generator[str, None, None]:
    """Dispatch streaming to the selected LLM provider."""
    if provider == 'ollama':
        yield from stream_ollama(prompt, model=model or DEFAULT_OLLAMA_MODEL)
    elif provider == 'openai':
        yield from stream_openai(
            prompt,
            model=model or 'gpt-4o-mini',
            api_key=api_key,
            max_tokens=max_tokens,
        )
    elif provider == 'anthropic':
        yield from stream_anthropic(
            prompt,
            model=model or 'claude-haiku-4-5-20251001',
            api_key=api_key,
            max_tokens=max_tokens,
        )
    elif provider == 'none':
        return
    else:
        raise ValueError(f'Unknown provider: {provider}')


def _single_chunk(text: str) -> Generator[str, None, None]:
    """Yield one fallback chunk."""
    yield text


def _prepend_chunk(
    first_chunk: str,
    gen: Generator[str, None, None],
) -> Generator[str, None, None]:
    """Yield an already-read chunk, then continue the original stream."""
    if first_chunk:
        yield first_chunk

    yield from gen


def stream_with_fallback(
    prompt: str,
    provider: str,
    model: str | None,
    fallback_answer: str,
    api_key: str | None = None,
) -> tuple[Generator[str, None, None], str]:
    """Return a streaming generator and the source that will answer."""
    if provider == 'none':
        return _single_chunk(fallback_answer), 'deterministic'

    try:
        primary = stream_llm(
            prompt,
            provider=provider,
            model=model,
            api_key=api_key,
        )
        first = next(primary)
        return _prepend_chunk(first, primary), provider
    except StopIteration:
        return _single_chunk(fallback_answer), 'deterministic'
    except Exception:
        pass

    if provider != 'ollama':
        try:
            fallback = stream_ollama(prompt, model=DEFAULT_OLLAMA_MODEL)
            first = next(fallback)
            return _prepend_chunk(first, fallback), 'ollama_fallback'
        except StopIteration:
            pass
        except Exception:
            pass

    return _single_chunk(fallback_answer), 'deterministic'


class UsageTracker:
    """Track approximate per-session hosted LLM usage and cost."""

    def __init__(self):
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        provider: str,
        model: str | None,
        prompt: str,
        response: str,
    ) -> None:
        self._records.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'provider': provider,
            'model': model,
            'input_tokens': estimate_tokens(prompt),
            'output_tokens': estimate_tokens(response),
            'estimated_cost_usd': estimate_cost(prompt, response, model),
        })

    def total_cost(self) -> float:
        return sum(float(row['estimated_cost_usd']) for row in self._records)

    def total_tokens(self) -> int:
        return sum(
            int(row['input_tokens']) + int(row['output_tokens'])
            for row in self._records
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def summary(self) -> str:
        calls = len(self._records)
        plural = 'call' if calls == 1 else 'calls'
        return f'{calls} {plural} · {self.total_tokens()} tokens · ${self.total_cost():.4f}'


def answer_with_optional_llm(
    question: str,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
    fallback_answer: str,
    use_llm: bool = False,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> tuple[str, bool]:
    """Return an LLM answer when available, otherwise return the fallback."""
    if not use_llm:
        return fallback_answer, False

    context = build_dataset_context(profile, prepared, results, feature_ranking)
    prompt = build_dataset_prompt(context, question)

    try:
        return ask_ollama(prompt, model=model), True
    except RuntimeError:
        return fallback_answer, False
