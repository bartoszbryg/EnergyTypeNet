import pandas as pd

from src.automl import prepare_dataset, profile_dataset, rank_features
from src.llm_assistant import (
    answer_with_optional_llm,
    build_dataset_context,
    build_dataset_prompt,
    parse_ollama_stream_line,
)


def test_build_dataset_context_includes_computed_results():
    df = pd.DataFrame({
        'signal': [0, 0, 1, 1, 2, 2],
        'target': ['low', 'low', 'mid', 'mid', 'high', 'high'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(df, 'target', ['signal'], 'classification')
    ranking = rank_features(prepared)
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'cv_accuracy': 0.91,
        'test_accuracy': 0.83,
        'test_f1_macro': 0.80,
    }])

    context = build_dataset_context(profile, prepared, results, ranking)

    assert 'Rows: 6' in context
    assert 'Task type: classification' in context
    assert 'Best baseline model: Logistic Regression' in context
    assert 'Evaluated baseline models: Logistic Regression' in context
    assert 'Baseline score table:' in context
    assert 'Test accuracy: 0.830' in context


def test_build_dataset_prompt_stays_grounded():
    prompt = build_dataset_prompt('Rows: 10', 'What matters most?')

    assert 'only use the computed facts' in prompt
    assert 'Do not suggest training models that are already listed as evaluated' in prompt
    assert 'Rows: 10' in prompt
    assert 'What matters most?' in prompt


def test_optional_llm_returns_fallback_when_disabled():
    answer, used_llm = answer_with_optional_llm(
        question='What is the best model?',
        profile={'n_rows': 1, 'n_columns': 1, 'missing_cells': 0, 'duplicate_rows': 0},
        prepared=None,
        results=None,
        feature_ranking=None,
        fallback_answer='fallback answer',
        use_llm=False,
    )

    assert answer == 'fallback answer'
    assert used_llm is False


def test_parse_ollama_stream_line_extracts_chunk_and_done_flag():
    chunk, done = parse_ollama_stream_line(
        b'{"response": "hello", "done": false}'
    )

    assert chunk == 'hello'
    assert done is False

    final_chunk, final_done = parse_ollama_stream_line(
        b'{"response": "", "done": true}'
    )

    assert final_chunk == ''
    assert final_done is True
