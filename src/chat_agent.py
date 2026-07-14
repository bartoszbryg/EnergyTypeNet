"""Multi-turn chat helpers for the AI Dataset Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.automl import PreparedDataset, answer_dataset_question
from src.llm_assistant import build_dataset_context


@dataclass
class ChatMessage:
    """One user or assistant message stored in dashboard session state."""

    role: str
    content: str
    source: str
    timestamp: str
    question_type: str
    grounded: bool


class ChatHistory:
    """Bounded chat history measured in user-assistant turns."""

    def __init__(self, max_turns: int = 20):
        self._messages: list[ChatMessage] = []
        self.max_turns = max_turns

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._trim()

    def last_n(self, n: int) -> list[ChatMessage]:
        return self._messages[-n:]

    def to_context_string(self, n: int = 6) -> str:
        labels = {
            'user': 'User',
            'assistant': 'Assistant',
        }
        lines = [
            f"{labels.get(message.role, message.role.title())}: {message.content}"
            for message in self.last_n(n * 2)
        ]

        return '\n'.join(lines)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [asdict(message) for message in self._messages]

    def clear(self) -> None:
        self._messages = []

    def user_questions(self) -> list[ChatMessage]:
        return [message for message in self._messages if message.role == 'user']

    def last_assistant_message(self) -> ChatMessage | None:
        for message in reversed(self._messages):
            if message.role == 'assistant':
                return message

        return None

    def _trim(self) -> None:
        max_messages = max(self.max_turns * 2, 2)

        while len(self._messages) > max_messages:
            if len(self._messages) >= 2 and self._messages[0].role == 'user':
                del self._messages[:2]
            else:
                del self._messages[0]


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for chat exports."""
    return datetime.utcnow().isoformat()


def classify_question(
    question: str,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
) -> str:
    """Classify a dataset question using transparent string rules."""
    del profile, prepared

    q = question.lower().strip()
    words = set(q.replace('?', ' ').replace(',', ' ').split())

    follow_up_starts = (
        'why',
        'how so',
        'what about',
        'can you',
        'tell me more',
        'explain',
        'elaborate',
    )
    pronoun_signals = {'that', 'it', 'this'}

    if q.startswith(follow_up_starts) or words.intersection(pronoun_signals):
        return 'follow_up'

    if any(signal in q for signal in ['vs', 'versus', 'compare', 'difference', 'better than']):
        return 'comparison'

    if any(signal in q for signal in ['should', 'recommend', 'suggest', 'would you']):
        return 'recommendation'

    if any(signal in q for signal in ['accuracy', 'f1', 'score', 'model', 'performance', 'best']):
        return 'model_performance'

    if any(signal in q for signal in ['feature', 'important', 'relevant', 'mutual', 'which column']):
        return 'feature_importance'

    if any(signal in q for signal in ['how many', 'rows', 'columns', 'missing', 'type', 'shape']):
        return 'dataset_stats'

    return 'general'


def build_contextualized_prompt(
    question: str,
    history: ChatHistory,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
    question_type: str,
) -> str:
    """Build a grounded LLM prompt that includes recent conversation history."""
    dataset_context = build_dataset_context(profile, prepared, results, feature_ranking)
    conversation_context = history.to_context_string(n=4)
    type_instructions = {
        'follow_up': (
            'The user is asking a follow-up question. Refer to the previous '
            'assistant answer in the conversation above.'
        ),
        'comparison': (
            'Compare the two things the user mentions specifically. Use only '
            'numbers from the context above.'
        ),
        'recommendation': (
            'Give a concrete recommendation based only on the statistics in '
            'the context. Do not hallucinate new numbers.'
        ),
    }
    instruction = type_instructions.get(
        question_type,
        'Answer the current dataset question from the computed facts.',
    )

    return (
        'You are an ML dataset assistant. Use the computed facts and the recent '
        'conversation only. It is allowed and expected to discuss model selection, '
        'feature importance, classification, regression, and evaluation metrics '
        'when those claims are grounded in the computed facts. If the evidence is '
        'not enough, say what should be computed next. You cannot execute code, '
        'train models, compute new metrics, access files, change dashboard state, '
        'or perform actions from this chat. If the user asks you to do something, '
        'do not refuse the ML topic; say you cannot run new computations from '
        'chat yet, summarize the existing computed results, and name the exact '
        'dashboard step or computation needed next. Do not say "I will compute", '
        '"I can perform", "I will run", or imply that new calculations were '
        'completed.\n\n'
        f'Computed facts:\n{dataset_context}\n\n'
        f'Recent conversation:\n{conversation_context or "No previous conversation."}\n\n'
        f'Current question: {question}\n\n'
        f'Instruction: {instruction}\n'
        'Answer concisely in 2-3 sentences. Ground every claim in the statistics '
        'shown in the context block above.'
    )


def handle_follow_up(
    question: str,
    history: ChatHistory,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
) -> str:
    """Answer follow-ups deterministically before falling back to base QA."""
    question_type = classify_question(question, profile, prepared)
    q = question.lower().strip()

    if question_type != 'follow_up':
        return answer_dataset_question(question, profile, prepared, results, feature_ranking)

    last_assistant = history.last_assistant_message()

    if last_assistant is None:
        return answer_dataset_question(question, profile, prepared, results, feature_ranking)

    if 'concrete example' in q or 'example' in q:
        if results is not None and not results.empty:
            best = results.iloc[0]

            if 'test_accuracy' in results.columns:
                return (
                    f"Concrete example: if you are choosing a deployable baseline, "
                    f"start with {best['model']} because it reached test accuracy "
                    f"{float(best['test_accuracy']):.3f} and macro F1 "
                    f"{float(best['test_f1_macro']):.3f}. Then compare it against "
                    "the dummy baseline to confirm it learned signal rather than class frequency."
                )

            return (
                f"Concrete example: if you need a regression baseline, start with "
                f"{best['model']} because it reached test R2 {float(best['test_r2']):.3f} "
                f"and test MAE {float(best['test_mae']):.3f}. Then inspect residuals "
                "before trusting it for high-stakes predictions."
            )

        if feature_ranking is not None and not feature_ranking.empty:
            top = feature_ranking.iloc[0]

            return (
                f"Concrete example: `{top['feature']}` is the strongest measured feature "
                f"with mutual information {float(top['mutual_information']):.4f}, so test "
                "a compact model that keeps it and removes weak features."
            )

    if 'what should i do' in q or 'what should' in q:
        if results is not None and not results.empty:
            best = results.iloc[0]

            return (
                f"Use {best['model']} as the first baseline, then validate whether a simpler "
                "compact feature set performs similarly. If the score gap is tiny, prefer the "
                "simpler feature set because it is easier to explain and maintain."
            )

        return (
            "Train the baseline models first, then compare the best model against the dummy "
            "baseline and the compact feature set before deciding what to keep."
        )

    if 'typical dataset' in q or 'compare to' in q:
        missing_cells = int(profile.get('missing_cells', 0))
        duplicate_rows = int(profile.get('duplicate_rows', 0))
        n_rows = int(profile.get('n_rows', 0))
        n_columns = int(profile.get('n_columns', 0))

        return (
            f"Compared with a typical messy CSV, this dataset looks clean on basic quality "
            f"checks: {n_rows:,} rows, {n_columns:,} columns, {missing_cells:,} missing cells "
            f"and {duplicate_rows:,} duplicate rows. The next question is not data cleanliness, "
            "but whether the selected features contain enough signal for the target."
        )

    if last_assistant.question_type == 'model_performance':
        if results is None or results.empty:
            return 'Train baseline models first so I can explain model performance in more detail.'

        best = results.iloc[0]
        second = results.iloc[1] if len(results) > 1 else None

        is_classification = (
            (prepared is not None and prepared.task_type == 'classification')
            or 'test_accuracy' in results.columns
        )

        if is_classification:
            answer = (
                f"{best['model']} is strongest because it has test accuracy "
                f"{float(best['test_accuracy']):.3f} and macro F1 "
                f"{float(best['test_f1_macro']):.3f}."
            )

            if second is not None:
                answer += (
                    f" The next model is {second['model']} with test accuracy "
                    f"{float(second['test_accuracy']):.3f}, so the gap is "
                    f"{float(best['test_accuracy'] - second['test_accuracy']):.3f}."
                )

            return answer

        answer = (
            f"{best['model']} is strongest because it has test R2 "
            f"{float(best['test_r2']):.3f} and test MAE {float(best['test_mae']):.3f}."
        )

        if second is not None:
            answer += (
                f" The next model is {second['model']} with test R2 "
                f"{float(second['test_r2']):.3f}."
            )

        return answer

    if last_assistant.question_type == 'feature_importance':
        if feature_ranking is None or feature_ranking.empty:
            return 'Run feature ranking first so I can explain the weakest and strongest columns.'

        weakest = ', '.join(feature_ranking.tail(3)['feature'].astype(str))
        strongest = ', '.join(feature_ranking.head(3)['feature'].astype(str))

        return (
            f"The strongest measured features are {strongest}. The weakest measured "
            f"features are {weakest}; compare model performance before dropping them."
        )

    if last_assistant.question_type == 'dataset_stats':
        column_rows = profile.get('columns', [])
        numeric = [
            row['column']
            for row in column_rows
            if 'int' in str(row.get('dtype', '')) or 'float' in str(row.get('dtype', ''))
        ]
        categorical = [
            row['column']
            for row in column_rows
            if row['column'] not in numeric
        ]

        return (
            f"The dataset has {len(numeric)} numeric columns and {len(categorical)} "
            f"categorical/text columns. Example numeric columns: {', '.join(numeric[:3]) or 'none'}; "
            f"example categorical columns: {', '.join(categorical[:3]) or 'none'}."
        )

    if last_assistant.question_type == 'comparison':
        return (
            'Looking at the other side of the comparison: prefer the option with '
            'the simpler feature set when performance is nearly tied, and prefer '
            'the higher-scoring option when the metric gap is meaningful.'
        )

    excerpt = last_assistant.content[:200].strip()

    return (
        f"Based on my previous answer: {excerpt}. Could you clarify what specific "
        "aspect you'd like me to expand on?"
    )


def suggest_next_questions(
    question_type: str,
    profile: dict[str, Any] | None,
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
) -> list[str]:
    """Return three context-aware follow-up questions."""
    profile = profile or {}
    best_model = 'the best model'
    second_model = 'the second-best model'

    if results is not None and not results.empty:
        best_model = str(results.iloc[0]['model'])

        if len(results) > 1:
            second_model = str(results.iloc[1]['model'])

    top_feature = 'the top feature'

    if prepared is not None and prepared.feature_cols:
        top_feature = str(prepared.feature_cols[0])

    target = 'the recommended target column'

    if prepared is not None:
        target = f'`{prepared.target_col}`'

    if question_type == 'dataset_stats':
        return [
            'Which features have the most missing values?',
            f'Why is {target} a good target?',
            'Are there any duplicate rows?',
        ]

    if question_type == 'model_performance':
        return [
            f'Why did {best_model} perform best?',
            'Which class is hardest to predict?',
            f'How does {best_model} compare to {second_model}?',
        ]

    if question_type == 'feature_importance':
        return [
            f'Why is {top_feature} important?',
            'Should I drop the least important features?',
            'How does feature selection affect accuracy?',
        ]

    if question_type == 'follow_up':
        return [
            'Can you give a concrete example?',
            'What should I do with this information?',
            'How does this compare to a typical dataset?',
        ]

    return [
        'What is the best target column?',
        'Which features look strongest?',
        'Which model should I trust most?',
    ]
