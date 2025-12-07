"""
Provider-specific prompt templates and utilities.
"""
from enum import Enum
from typing import Dict, Any


class JudgePromptTemplate:
    """Templates for judge LLM prompts."""

    @staticmethod
    def accuracy_judge(question: str, reference_answer: str, model_answer: str) -> str:
        """Prompt for judging accuracy on a scale of 0-1."""
        return f"""
You are an impartial judge evaluating the quality of an AI model's answer to a question.

**Question:** {question}

**Reference Answer (ground truth):** {reference_answer}

**Model's Answer:** {model_answer}

Please evaluate the model's answer for correctness and quality.
Output a single floating-point number between 0.0 and 1.0, where 1.0 means perfectly correct and 0.0 means completely incorrect.
Also provide a brief explanation (one sentence) after the score.

Format your response exactly as:
SCORE: <number>
EXPLANATION: <text>
"""

    @staticmethod
    def categorical_judge(question: str, reference_answer: str, model_answer: str, categories: list) -> str:
        """Prompt for categorical judgment."""
        categories_str = ", ".join(categories)
        return f"""
You are an impartial judge evaluating the quality of an AI model's answer to a question.

**Question:** {question}

**Reference Answer (ground truth):** {reference_answer}

**Model's Answer:** {model_answer}

Please categorize the model's answer into one of the following categories: {categories_str}.
Output the category name only.
"""


class TestPromptTemplate:
    """Templates for test LLM prompts."""

    @staticmethod
    def answer_question(question: str) -> str:
        """Prompt for answering a question."""
        return f"""
Please answer the following question to the best of your ability.

Question: {question}

Provide a clear, step-by-step reasoning followed by the final answer.
"""