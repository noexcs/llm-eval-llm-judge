"""
GSM8K task implementation.
"""
import logging
import re
from platform import system
from typing import List, Dict, Literal

from openai.types.chat import ChatCompletionUserMessageParam, \
    ChatCompletionSystemMessageParam

from llm_eval.tasks.base import Task, TaskSample, Judgment, Metrics, Prompt, JudgePolicy

logger = logging.getLogger(__name__)


class GS8MKTask(Task):
    """Grade School Math 8K task."""

    def __init__(self):
        self.loader = None
        self.split = "test"
        self.limit = None

    def judge_policy(self) -> JudgePolicy:
        return JudgePolicy.LLM

    def load_samples(self) -> List[TaskSample]:
        """Load GSM8K dataset via Hugging Face datasets library."""
        try:
            import datasets
        except ImportError:
            raise ImportError(
                "Hugging Face datasets library is required to load GSM8K. "
                "Install with `pip install datasets` or `pip install llm-eval[datasets]`."
            )
        try:
            dataset = datasets.load_dataset("gsm8k", split=self.split)
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8K dataset from Hugging Face: {e}")

        samples: List[TaskSample] = []
        for i, item in enumerate(dataset):
            if self.limit is not None and len(samples) >= self.limit:
                break
            # GSM8K dataset has 'question' and 'answer' fields
            question = item.get("question", "")
            answer = item.get("answer", "")
            samples.append(
                dict(
                    id=f"gsm8k_{self.split}_{i}",
                    question=question,
                    reference_answer=answer,
                    metadata={"source": "gsm8k", "split": self.split, "via": "hf_datasets"},
                )
            )
        logger.info(f"Loaded {len(samples)} samples from Hugging Face GSM8K {self.split} split")
        return samples


    def build_test_prompt(self, sample: Dict) -> Prompt:
        """Build prompt for test LLM."""
        prompt_text = (
            f"Question: {sample['question']}\n\n"
            "Provide your final answer as a single number (or a simple expression) inside a boxed \\boxed{}.\n"
        )
        return [
            ChatCompletionSystemMessageParam(role="system", content="You're helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content=prompt_text)
        ]

    def build_judge_prompt(self, sample: Dict, test_response: str) -> Prompt:
        """Build prompt for judge LLM."""
        system_prompt = (
            "You are a strict but fair math evaluator. Your task is to determine whether the model's final answer "
            "is mathematically equivalent to the reference answer.\n\n"
            "Rules:\n"
            "- Focus ONLY on the final numerical value (ignore units, text, or intermediate steps).\n"
            "- Extract the final answer from the model's response. Look for numbers inside \\boxed{}, "
            "or the last number mentioned if no box is present.\n"
            "- Accept equivalent forms: integers, decimals, fractions, percentages (e.g., 0.5 = 1/2 = 50%).\n"
            "- Ignore commas used as thousand separators (1,234 = 1234).\n"
            "- The reference answer is provided both in raw form and as a cleaned numerical value for comparison.\n"
            "- Output exactly one score: 1 if numerically equivalent, 0 otherwise.\n\n"
            "Respond in the following format only:\n"
            "SCORE: 1  or  SCORE: 0\n"
            "EXPLANATION: <optional very short explanation only if score is 0>\n"
        )
        prompt_text = (
            f"Question: {sample['question']}\n\n"
            f"Reference Answer (ground truth): {sample['reference_answer']}\n\n"
            f"Model's answer: {test_response}\n\n"
        )
        return [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=prompt_text)
        ]

    def parse_judgment(self, judge_response: str) -> Judgment:
        """Parse raw judge LLM response into a structured Judgment."""
        # Extract score using regex
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", judge_response, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Fallback: look for any number
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", judge_response)
            if numbers:
                score = float(numbers[0])
            else:
                score = 0.0

        # Extract explanation
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", judge_response, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return dict(
            score=score,
            explanation=explanation,
        )

    def rule_based_judge(self, sample: Dict, test_response: str) -> Judgment:
        """Rule-based judge."""
        return dict(
            score=0,
            explanation=None,
        )

    def calculate_metrics(self, judgments: List[Judgment]) -> Metrics:
        """Calculate accuracy and other metrics."""
        if not judgments:
            return dict(
                num_samples=0,
                average_score=0.0,
                median_score=0.0,
                std_score=0.0,
                accuracy=0.0,
                pass_rate=0.0,
                score_distribution={},
            )
        scores = [j['score'] for j in judgments]
        import numpy as np
        average_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        accuracy = np.mean([1 if s >= 0.5 else 0 for s in scores])
        pass_rate = np.mean([1 if s >= 0.8 else 0 for s in scores])
        # Create distribution bins
        bins = {"0": 0, "1": 0}
        for s in scores:
            if s < 0.5:
                bins["0"] += 1
            else:
                bins["1"] += 1
        return dict(
            num_samples=len(judgments),
            average_score=average_score,
            median_score=median_score,
            std_score=std_score,
            accuracy=accuracy * 100,  # percentage
            pass_rate=pass_rate * 100,
            score_distribution=bins,
        )

    @property
    def name(self) -> str:
        return "gsm8k"
