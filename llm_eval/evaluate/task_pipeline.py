"""
Task‑based evaluation pipeline.
"""
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, TypedDict

from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm
from dataclasses import dataclass

from llm_eval.config import EvaluationConfig, EvaluationMode
from llm_eval.llm.client import create_llm_client
from llm_eval.tasks.base import Task, TaskSample, Prompt, Judgment, Metrics, JudgePolicy
from llm_eval.tasks import get_task

logger = logging.getLogger(__name__)


def batch(iterable, n=1):
    """Yield successive n-sized chunks from iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


@dataclass
class Result(TypedDict):
    sample: TaskSample
    judgment: Judgment

    test_response: str
    test_prompt: Prompt

    judge_response: Optional[str]
    judge_prompt: Optional[Prompt]

MAX_REJUDGE_TRIES = 3

class TaskPipeline:
    """Task‑based evaluation pipeline."""

    def __init__(self, config: EvaluationConfig, task: Task):
        self.config = config
        self.task = task

        self.test_client = None
        if config.mode in [EvaluationMode.UNIFIED, EvaluationMode.ANSWER]:
            self.test_client = create_llm_client(config.test_llm)

        self.judge_client = None
        if config.mode in [EvaluationMode.UNIFIED, EvaluationMode.JUDGE]:
            self.judge_client = create_llm_client(config.judge_llm)

        self.samples: Optional[List[TaskSample]] = None

    def load_samples(self):
        """Load dataset samples using the task."""
        logger.info(f"Loading dataset for task {self.task.name}...")
        self.samples = self.task.load_samples()
        logger.info(f"Loaded {len(self.samples)} samples.")

    def run_unified(self) -> List[Result]:
        """Run unified evaluation (answer and judge in one go) with batching."""
        if self.samples is None:
            self.load_samples()
        if self.judge_client is None:
            raise RuntimeError("Judge client not initialized for unified mode.")

        results = []
        batch_size = self.config.test_llm.batch_size
        total_batches = (len(self.samples) + batch_size - 1) // batch_size
        for batch_samples in tqdm(batch(self.samples, batch_size), desc="Unified evaluation", total=total_batches):
            # Build test prompts
            test_prompts: Iterable[Iterable[ChatCompletionMessageParam]] = [self.task.build_test_prompt(sample)
                                                                            for sample in batch_samples]
            try:
                test_responses = self.test_client.generate_batch(prompts=test_prompts)
            except Exception as e:
                logger.error(f"Failed to answer batch: {e}")
                test_responses = [""] * len(batch_samples)

            # Build judge prompts
            judge_prompts: Iterable[Iterable[ChatCompletionMessageParam]] = [
                self.task.build_judge_prompt(sample, test_response)
                for sample, test_response in zip(batch_samples, test_responses)
            ]
            try:
                judge_responses = self.judge_client.generate_batch(prompts=judge_prompts)
            except Exception as e:
                logger.error(f"Failed to judge batch: {e}")
                judge_responses = [""] * len(batch_samples)

            # Parse judgments
            for sample, test_prompt, test_response, judge_prompt, judge_response in zip(
                    batch_samples, test_prompts, test_responses, judge_prompts, judge_responses
            ):
                judgment = self.task.parse_judgment(judge_response)
                if judgment['rejudge']:
                    for _ in range(MAX_REJUDGE_TRIES):
                        if not judgment.get('rejudge', False):
                            break
                        judge_response = self.judge_client.generate_batch([judge_prompt], temperature=random.uniform(0, 0.5))[0]
                        judgment = self.task.parse_judgment(judge_response)
                    if judgment['rejudge']:
                        logger.error(
                            f"Following judge failed after {MAX_REJUDGE_TRIES} rejudges.\n\n"
                            f"Sample:\n {sample}\n\n"
                            f"Test response:\n {test_response}\n\n"
                            f"Final judge response:\n {judge_response}"
                        )

                results.append({
                    "sample": sample,

                    "judgment": judgment,

                    "test_response": test_response,
                    "judge_response": judge_response,

                    "test_prompt": test_prompt,
                    "judge_prompt": judge_prompt,
                })

        return results

    def run_answer(self) -> List[Result]:
        """Generate answers only."""
        if self.samples is None:
            self.load_samples()

        answers = []
        batch_size = self.config.test_llm.batch_size
        for batch_samples in batch(self.samples, batch_size):
            test_prompts: Iterable[Iterable[ChatCompletionMessageParam]] = [self.task.build_test_prompt(sample)
                                                                            for sample in batch_samples]
            try:
                test_responses = self.test_client.generate_batch(prompts=test_prompts)
            except Exception as e:
                logger.error(f"Failed to answer batch: {e}")
                test_responses = [""] * len(batch_samples)

            for sample, prompt, response in zip(batch_samples, test_prompts, test_responses):
                answers.append({
                    "sample": sample,
                    "test_response": response,
                    "test_prompt": prompt,
                })

        return answers

    def run_judge(self, answers_file: Path) -> List[Result]:
        """Judge pre‑answered from file."""
        if self.judge_client is None:
            raise RuntimeError("Judge client not initialized for judge mode.")

        with open(answers_file, "r", encoding="utf-8") as f:
            answers = [json.loads(line) for line in f]

        logger.info(f"Judge policy: {self.task.judge_policy}")

        if self.task.judge_policy == JudgePolicy.RULE:
            return self.rule_judge(answers)

        if self.task.judge_policy == JudgePolicy.LLM:
            return self.llm_judge(answers)

        if self.task.judge_policy == JudgePolicy.RULE_AND_LLM:
            r =  self.rule_judge(answers)
            not_rejudges = [item for item in r if not item["judgment"]['rejudge']]
            rejudges = [item for item in r if item["judgment"]['rejudge']]
            rejudge_results = self.llm_judge(rejudges)
            return not_rejudges + rejudge_results

        if self.task.judge_policy == JudgePolicy.LLM_AND_RULE:
            r = self.llm_judge(answers)
            not_rejudges = [item for item in r if not item["judgment"]['rejudge']]
            rejudges = [item for item in r if item["judgment"]['rejudge']]
            rejudge_results = self.rule_judge(rejudges)
            return not_rejudges + rejudge_results

        return []

    def rule_judge(self, answers: list[Any]) -> list[Any]:
        results = []
        batch_size = self.config.judge_llm.batch_size
        for batch_answer_samples in batch(answers, batch_size):
            for answer_sample in batch_answer_samples:
                judgment = self.task.rule_based_judge(answer_sample["sample"], answer_sample["test_response"])
                results.append({
                    "sample": answer_sample["sample"],
                    "judgment": judgment,
                    "test_response": answer_sample["test_response"],
                    "test_prompt": answer_sample["test_prompt"],
                })
        return results

    def llm_judge(self, answers: list[Any]) -> list[Any]:
        results = []
        batch_size = self.config.judge_llm.batch_size
        for batch_answer_samples in batch(answers, batch_size):
            judge_prompts: Iterable[Iterable[ChatCompletionMessageParam]] = [
                self.task.build_judge_prompt(answer_sample["sample"], answer_sample["test_response"])
                for answer_sample in batch_answer_samples
            ]
            try:
                judge_responses = self.judge_client.generate_batch(prompts=judge_prompts)
            except Exception as e:
                logger.error(f"Failed to judge batch: {e}")
                judge_responses = [""] * len(batch_answer_samples)

            for answer_sample, judge_prompt, judge_response in zip(batch_answer_samples,
                                                                   judge_prompts,
                                                                   judge_responses):
                judgment = self.task.parse_judgment(judge_response)
                if judgment['rejudge']:
                    for _ in range(MAX_REJUDGE_TRIES):
                        if not judgment.get('rejudge', False):
                            break

                        judge_response = self.judge_client.generate_batch([judge_prompt],
                                                                          temperature=random.uniform(0, 0.5))[0]
                        judgment = self.task.parse_judgment(judge_response)
                    if judgment['rejudge']:
                        logger.error(
                            f"Following judge failed after {MAX_REJUDGE_TRIES} rejudges.\n\n"
                            f"Sample:\n {answer_sample['sample']}\n\n"
                            f"Test response:\n {answer_sample['test_response']}\n\n"
                            f"Final judge response:\n {judge_response}"
                        )

                results.append({
                    "sample": answer_sample["sample"],

                    "judgment": judgment,

                    "test_response": answer_sample["test_response"],
                    "judge_response": judge_response,

                    "test_prompt": answer_sample["test_prompt"],
                    "judge_prompt": judge_prompt,
                })
        return results

    @staticmethod
    def save_answers(answers: List[Result], path: Path):
        """Save generated answers to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for item in answers:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(answers)} answers to {path}")

    @staticmethod
    def save_results(results: List[Result], path: Path, metrics: Optional[Metrics] = None):
        """Save evaluation results to JSON file."""
        if metrics is not None:
            data = {"metrics": metrics, "results": results}
        else:
            data = results
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results to {path}")


def run_task_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """High‑level task‑based evaluation runner."""

    task_name = config.task_name
    task = get_task(config.task_name)

    pipeline = TaskPipeline(config, task)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    match config.mode:
        case config.mode.UNIFIED:
            results = pipeline.run_unified()
            metrics = task.calculate_metrics([
                r['judgment'] for r in results
            ])

            output_path = config.output.output_dir / f"{config.test_llm.model}-{task_name}-{timestamp}.json"
            TaskPipeline.save_results(results, output_path, metrics=metrics)
            return {"results": results, "metrics": metrics, "output_path": output_path}

        case config.mode.ANSWER:
            answers = pipeline.run_answer()

            output_path = config.output.output_dir / f"{config.test_llm.model}-{task_name}-ANSWER-{timestamp}.jsonl"
            TaskPipeline.save_answers(answers, output_path)
            return {"answers": answers, "output_path": output_path}

        case config.mode.JUDGE:
            if not config.output.answers_file:
                raise ValueError("Answers file must be provided for judge mode.")
            results = pipeline.run_judge(config.output.answers_file)
            metrics = task.calculate_metrics([
                r['judgment'] for r in results
            ])

            answers_file_name = config.output.answers_file.stem
            output_path = config.output.output_dir / f"{answers_file_name}-JUDGE-{config.judge_llm.model}-{timestamp}.json"
            TaskPipeline.save_results(results, output_path, metrics=metrics)
            return {"results": results, "metrics": metrics, "output_path": output_path}

        case _:
            raise ValueError(f"Invalid mode: {config.mode}")
