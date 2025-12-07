"""
Command-line interface for LLM evaluation.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from llm_eval.config import (
    EvaluationConfig,
    LLMConfig,
    OutputConfig,
    EvaluationMode,
    LLMProvider,
)
from llm_eval.evaluate.task_pipeline import run_task_evaluation


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation CLI: Comprehensive LLM task evaluation with dual-LLM architecture."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unified", "answer", "judge"],
        default="unified",
        help="Evaluation mode: unified (answer+judge), answer (answers only), judge (judge pre-answered answers)",
    )

    # Test LLM arguments
    parser.add_argument(
        "--test-api",
        type=str,
        choices=["openai", "anthropic", "openai-compatible"],
        required=False,
        help="API provider for test LLM (required for unified and answer modes)",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        required=False,
        help="Model identifier for test LLM (required for unified and answer modes)",
    )
    parser.add_argument(
        "--test-api-key",
        type=str,
        help="API key for test LLM (optional, can be set via environment variable)",
    )
    parser.add_argument(
        "--test-api-base",
        type=str,
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Batch size for concurrent test LLM requests (default: 1)",
    )
    parser.add_argument(
        "--test-temperature",
        type=float,
        default=0.0,
        help="Temperature for test LLM (default: 0.0)",
    )
    parser.add_argument(
        "--test-max-retries",
        type=int,
        default=3,
        help="Maximum retries for test LLM (default: 3)",
    )

    # Judge LLM arguments (optional for answer mode)
    parser.add_argument(
        "--judge-api",
        type=str,
        choices=["openai", "anthropic", "openai-compatible"],
        required=False,
        help="API provider for judge LLM (required for unified and judge modes)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        required=False,
        help="Model identifier for judge LLM (required for unified and judge modes)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        help="API key for judge LLM (optional, can be set via environment variable)",
    )
    parser.add_argument(
        "--judge-api-base",
        type=str,
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=1,
        help="Batch size for concurrent judge LLM requests (default: 1)",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judge LLM (default: 0.0)",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=3,
        help="Maximum retries for judge LLM (default: 3)",
    )

    # Tasks arguments
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of task names (e.g., gsm8k,math).",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for all outputs (default: ./results)",
    )
    parser.add_argument(
        "--answers-file",
        type=Path,
        help="Path to pre-answered answers for judge mode",
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Validate arguments based on mode

    # Tasks required
    if not args.tasks:
        parser.error("--tasks is required")

    if args.mode == "unified":
        # Test LLM required
        if not args.test_api:
            parser.error(f"--test-api is required for mode {args.mode}")
        if not args.test_model:
            parser.error(f"--test-model is required for mode {args.mode}")
        # Judge LLM required
        if not args.judge_api:
            parser.error(f"--judge-api is required for mode {args.mode}")
        if not args.judge_model:
            parser.error(f"--judge-model is required for mode {args.mode}")
    if args.mode == "answer":
        # Test LLM required
        if not args.test_api:
            parser.error(f"--test-api is required for mode {args.mode}")
        if not args.test_model:
            parser.error(f"--test-model is required for mode {args.mode}")
    if args.mode == "judge":
        # Judge LLM required
        if not args.judge_api:
            parser.error(f"--judge-api is required for mode {args.mode}")
        if not args.judge_model:
            parser.error(f"--judge-model is required for mode {args.mode}")
        # Answers file required
        if not args.answers_file:
            parser.error("--answers-file is required for judge mode")

    return args


def create_config_for_task(args, task_name, output_dir, answers_file=None):
    """Create EvaluationConfig."""

    # Mode
    mode_map = {
        "unified": EvaluationMode.UNIFIED,
        "answer": EvaluationMode.ANSWER,
        "judge": EvaluationMode.JUDGE,
    }
    mode = mode_map[args.mode]

    # Test LLM config (same as original)
    test_llm = None
    if mode in [EvaluationMode.UNIFIED, EvaluationMode.ANSWER]:
        test_llm = LLMConfig(
            provider=LLMProvider(args.test_api),
            model=args.test_model,
            api_key=args.test_api_key,
            api_base=args.test_api_base,
            temperature=args.test_temperature,
            batch_size=args.test_batch_size,
        )

    # Judge LLM config
    judge_llm = None
    if mode in [EvaluationMode.UNIFIED, EvaluationMode.JUDGE]:
        judge_llm = LLMConfig(
            provider=LLMProvider(args.judge_api),
            model=args.judge_model,
            api_key=args.judge_api_key,
            api_base=args.judge_api_base,
            temperature=args.judge_temperature,
            batch_size=args.judge_batch_size,
        )

    # Output config
    output = OutputConfig(
        output_dir=output_dir,
        answers_file=answers_file,
    )

    return EvaluationConfig(
        mode=mode,
        task_name=task_name,
        test_llm=test_llm,
        judge_llm=judge_llm,
        output=output,
        seed=args.seed,
        log_level=args.log_level,
    )


def main():
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    # Determine task list
    task_names = [t.strip() for t in args.tasks.split(",")]

    # For each task, run evaluation
    for task_name in task_names:
        logging.info(f"Processing task: {task_name}")

        # Create output directory per task: output_dir / task_name
        task_output_dir = args.output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        answers_file = None
        if args.answers_file:
            # For judge mode.
            answers_file = args.answers_file

        try:
            config = create_config_for_task(args, task_name, task_output_dir, answers_file)
        except Exception as e:
            logging.error(f"Invalid configuration for task '{task_name}': {e}")
            sys.exit(1)

        logging.info(f"Starting evaluation in {config.mode} mode...")
        try:
            run_task_evaluation(config)
            logging.info(f"Task '{task_name}' evaluation completed successfully.")
        except Exception as e:
            logging.error(f"Evaluation failed for task '{task_name}': {e}", exc_info=True)
            sys.exit(1)


    logging.info("All tasks evaluation completed successfully.")


if __name__ == "__main__":
    main()
