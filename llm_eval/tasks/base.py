"""
Base task class for LLM evaluation.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Literal, Iterable
from pathlib import Path
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

TaskSample = Dict[str, Any]
Judgment = Dict[str, Any]
Metrics = Dict[str, Any]

Prompt = Iterable[ChatCompletionMessageParam]
Prompts = Iterable[Prompt]


class JudgePolicy(str, Enum):
    LLM = "llm"
    RULE = "rule"


class Task(ABC):
    """Abstract base class for an evaluation task."""

    @property
    @abstractmethod
    def judge_policy(self) -> JudgePolicy:
        """Judge policy for the task - either 'llm' or 'rule'."""
        pass

    @abstractmethod
    def load_samples(self) -> List[TaskSample]:
        """Load dataset from path and return list of TaskSample."""
        pass

    @abstractmethod
    def build_test_prompt(self, sample: Dict) -> Prompt:
        """Build prompt for test LLM given a sample."""
        pass

    @abstractmethod
    def build_judge_prompt(self, sample: Dict, test_response: str) -> Prompt:
        """Build prompt for judge LLM given sample and test LLM response."""
        pass

    @abstractmethod
    def parse_judgment(self, judge_response: str) -> Judgment:
        """Parse raw judge LLM response into a structured Judgment."""
        pass

    @abstractmethod
    def rule_based_judge(self, sample: Dict, model_response: str) -> Judgment:
        """Rule-based judge for tasks without a judge LLM."""
        pass

    @abstractmethod
    def calculate_metrics(self, judgments: List[Judgment]) -> Metrics:
        """Calculate task-specific metrics from a list of judgments."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the task."""
        pass
