"""Optional local LLM explanations for dataset assistant results."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

from src.automl import PreparedDataset


DEFAULT_OLLAMA_URL = 'http://localhost:11434/api/generate'
DEFAULT_OLLAMA_MODEL = 'llama3.1'


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
        'computed facts below. If the facts are not enough, say what should be '
        'computed next. Do not suggest training models that are already listed '
        'as evaluated. Keep the answer concise and practical.\n\n'
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
