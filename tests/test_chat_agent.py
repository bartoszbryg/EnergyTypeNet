import pandas as pd

from src.automl import prepare_dataset, profile_dataset, rank_features
from src.chat_agent import (
    ChatHistory,
    ChatMessage,
    build_contextualized_prompt,
    classify_question,
    handle_follow_up,
    suggest_next_questions,
)


def test_classify_question_follow_up():
    assert classify_question('why is that?', {}, None) == 'follow_up'


def test_classify_question_model_performance():
    assert classify_question('which model has the best accuracy?', {}, None) == 'model_performance'


def test_chat_message_generates_timestamp_when_omitted():
    message = ChatMessage(role='user', content='hello')

    assert message.timestamp
    assert message.source == 'user'
    assert message.question_type == 'general'
    assert message.grounded is False


def test_classify_question_computation_request():
    assert classify_question('Can you compute model complexity?', {}, None) == 'computation'


def test_classify_question_prompt_keyword_coverage():
    assert classify_question('Show prediction results', {}, None) == 'model_performance'
    assert classify_question('Which predictor is strongest?', {}, None) == 'feature_importance'
    assert classify_question('Are there duplicate or null values?', {}, None) == 'dataset_stats'
    assert classify_question('Which is better than the other?', {}, None) == 'comparison'
    assert classify_question('How should I use this model?', {}, None) == 'recommendation'


def test_chat_history_max_turns():
    history = ChatHistory(max_turns=4)

    for i in range(6):
        history.add(ChatMessage('user', f'q{i}', 'user', '', 'general', False))
        history.add(ChatMessage('assistant', f'a{i}', 'deterministic', '', 'general', True))

    assert len(history.messages) <= 8
    assert history.messages[0].content == 'q2'


def test_chat_history_context_string():
    history = ChatHistory()
    history.add(ChatMessage('user', 'hello', 'user', '', 'general', False))
    history.add(ChatMessage('assistant', 'hi there', 'deterministic', '', 'general', True))

    context = history.to_context_string(n=2)

    assert 'User: hello' in context
    assert 'Assistant: hi there' in context


def test_to_dict_list():
    history = ChatHistory()
    history.add(ChatMessage('user', 'q', 'user', '2025-01-01', 'general', False))

    data = history.to_dict_list()

    assert data[0]['role'] == 'user'
    assert data[0]['content'] == 'q'


def test_suggest_next_questions_returns_three():
    suggestions = suggest_next_questions('dataset_stats', {}, None, None)

    assert len(suggestions) == 3
    assert all(isinstance(suggestion, str) for suggestion in suggestions)


def test_contextual_prompt_includes_history_and_instruction():
    df = pd.DataFrame({
        'feature': [0, 1, 2, 3, 4, 5],
        'target': ['a', 'a', 'b', 'b', 'c', 'c'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(df, 'target', ['feature'], 'classification')
    ranking = rank_features(prepared)
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'cv_accuracy': 0.9,
        'test_accuracy': 0.8,
        'test_f1_macro': 0.75,
    }])
    history = ChatHistory()
    history.add(ChatMessage('user', 'What is the best model?', 'user', '', 'model_performance', False))
    history.add(ChatMessage('assistant', 'Logistic Regression is best.', 'deterministic', '', 'model_performance', True))

    prompt = build_contextualized_prompt(
        'Why is that?',
        history,
        profile,
        prepared,
        results,
        ranking,
        'follow_up',
        tool_result='Tool result: Logistic Regression and KNN are tied.',
    )

    assert 'Recent conversation:' in prompt
    assert 'Logistic Regression is best.' in prompt
    assert 'follow-up question' in prompt
    assert 'Test accuracy: 0.800' in prompt
    assert 'allowed and expected to discuss model selection' in prompt
    assert 'Do not refuse the ML topic' in prompt
    assert 'use the tool result' in prompt
    assert 'Do not imply that calculations outside the provided tool result' in prompt
    assert 'Computed tool result:' in prompt
    assert 'Logistic Regression and KNN are tied' in prompt


def test_contextual_prompt_model_performance_instruction_mentions_metrics():
    df = pd.DataFrame({
        'feature': [0, 1, 2, 3, 4, 5],
        'target': ['a', 'a', 'b', 'b', 'c', 'c'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(df, 'target', ['feature'], 'classification')

    prompt = build_contextualized_prompt(
        'Which model performs best?',
        ChatHistory(),
        profile,
        prepared,
        pd.DataFrame(),
        pd.DataFrame(),
        'model_performance',
    )

    assert 'Cite specific accuracy' in prompt
    assert 'F1' in prompt


def test_handle_follow_up_uses_previous_answer_type():
    history = ChatHistory()
    history.add(ChatMessage('user', 'What is the best model?', 'user', '', 'model_performance', False))
    history.add(ChatMessage('assistant', 'Logistic Regression is best.', 'deterministic', '', 'model_performance', True))
    results = pd.DataFrame([
        {
            'model': 'Logistic Regression',
            'cv_accuracy': 0.9,
            'test_accuracy': 0.8,
            'test_f1_macro': 0.75,
        },
        {
            'model': 'Random Forest',
            'cv_accuracy': 0.85,
            'test_accuracy': 0.7,
            'test_f1_macro': 0.65,
        },
    ])

    answer = handle_follow_up('why is that?', history, {}, None, results, None)

    assert 'Logistic Regression' in answer
    assert '0.800' in answer


def test_handle_follow_up_concrete_example_from_results():
    history = ChatHistory()
    history.add(ChatMessage('user', 'What is the best model?', 'user', '', 'model_performance', False))
    history.add(ChatMessage('assistant', 'Logistic Regression is best.', 'deterministic', '', 'model_performance', True))
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'test_accuracy': 1.0,
        'test_f1_macro': 1.0,
    }])

    answer = handle_follow_up(
        'Can you give a concrete example?',
        history,
        {},
        None,
        results,
        None,
    )

    assert 'Concrete example' in answer
    assert 'Logistic Regression' in answer


def test_handle_follow_up_typical_dataset_uses_profile_stats():
    history = ChatHistory()
    history.add(ChatMessage('user', 'Does this dataset have missing values?', 'user', '', 'dataset_stats', False))
    history.add(ChatMessage('assistant', 'No missing cells.', 'deterministic', '', 'dataset_stats', True))
    profile = {
        'n_rows': 36,
        'n_columns': 9,
        'missing_cells': 0,
        'duplicate_rows': 0,
    }

    answer = handle_follow_up(
        'How does this compare to a typical dataset?',
        history,
        profile,
        None,
        None,
        None,
    )

    assert '36' in answer
    assert '0 missing cells' in answer
