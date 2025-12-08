"""
GSM8K task implementation.
"""
import logging
import re
from typing import List, Dict

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

    @property
    def judge_policy(self) -> JudgePolicy:
        return JudgePolicy.RULE_AND_LLM

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
"""
You are a strict but fair math evaluator. Your task is to determine whether the model's final answer is mathematically equivalent to the reference answer.

IMPORTANT RULES:
- DO NOT solve or re-derive the problem yourself.
- Extract the final answer from the model's answer. Look for numbers inside \\boxed{}, or the last number mentioned if no box is present.
- Focus ONLY on the final numerical value (ignore units, text, or intermediate steps).
- Accept equivalent forms: integers, decimals, fractions, percentages (e.g., 0.5 = 1/2 = 50%).
- Ignore commas used as thousand separators (1,234 = 1234).
- The reference answer is provided both in raw form and as a cleaned numerical value for comparison.
- Output exactly one score: 1 if numerically equivalent, 0 otherwise.

Respond in the following format only:
SCORE: 1  or  SCORE: 0

EXAMPLE 1:
# **Question**:
 What is the square root of 16?

# **Model's answer**:
 \\boxed{4}

# **Reference Answer (ground truth)**:
 \\boxed{4}

# **Output**:
The model's answer \\boxed{4} and the reference answer \\boxed{4} are mathematically equivalent. Correct. SCORE: 1

EXAMPLE 2:
# **Question**:
What is the answer of 1+1?

# **Model's answer**:
\\boxed{3}

# **Reference Answer (ground truth)**:
\\boxed{2}

# **Output**:
The model's answer \\boxed{3} and the reference answer \\boxed{2} are not mathematically equivalent. Incorrect. SCORE: 0

"""
        )
        prompt_text = (
f""""
# **Question**:
{sample['question']}

# **Model's answer**:
{test_response}

# **Reference Answer (ground truth)**:
{sample['reference_answer']}

# **Output**:
"""
        )
        return [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=prompt_text)
        ]

    def parse_judgment(self, judge_response: str) -> Judgment:
        score_matches = re.findall(r"SCORE:\s*(0|1)\s*", judge_response.strip(), re.IGNORECASE | re.MULTILINE)
        if len(score_matches) == 1:
            score = float(score_matches[0])
            return dict(
                score=score,
                rejudge=False,
            )
        else:
            return dict(score=0, rejudge=True)

    def rule_based_judge(self, sample: Dict, test_response: str) -> Judgment:
        """Rule-based judge."""
        boxed_matches = re.findall(r"boxed\{([^}]*)\}", test_response.strip(), re.IGNORECASE | re.MULTILINE)
        ground_truth_matches = re.findall(r"####\s*(.+)", sample['reference_answer'].strip(), re.IGNORECASE | re.MULTILINE)

        if len(boxed_matches) == 0 or len(ground_truth_matches) == 0:
            return dict(
                score=0,
                rejudge=True,
                test_ans=boxed_matches[-1].strip() if len(boxed_matches) > 0 else "",
                ground_truth_ans=ground_truth_matches[-1].strip() if len(ground_truth_matches) > 0 else ""
            )

        test_ans = boxed_matches[-1].strip()
        ground_truth_ans = ground_truth_matches[-1].strip()
        
        # Extract numeric values for comparison
        test_num = re.sub(r'[^\d\.\/\-+]', '', test_ans)
        ground_truth_num = re.sub(r'[^\d\.\/\-+]', '', ground_truth_ans)
        
        # Handle fraction conversion
        def eval_math_expr(expr):
            try:
                # Simple evaluation for fractions and basic math expressions
                if '/' in expr and expr.count('/') == 1:
                    numerator, denominator = expr.split('/')
                    return float(numerator) / float(denominator)
                else:
                    return float(expr)
            except:
                return None
                
        test_value = eval_math_expr(test_num)
        ground_truth_value = eval_math_expr(ground_truth_num)
        
        if test_value is not None and ground_truth_value is not None and abs(test_value - ground_truth_value) < 1e-9:
            return dict(
                score=1,
                rejudge=False,
                test_ans=test_ans,
                ground_truth_ans=ground_truth_ans
            )
        return dict(
            score=0,
            rejudge=False,
            test_ans=test_ans,
            ground_truth_ans=ground_truth_ans
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
        judge_failures = len([j for j in judgments if j['rejudge']])
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
            judge_failures=judge_failures,
        )

    @property
    def name(self) -> str:
        return "gsm8k"
