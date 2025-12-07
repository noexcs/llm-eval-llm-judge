# LLM Evaluation CLI

A production-grade Python command-line interface (CLI) tool for comprehensive LLM task evaluation using a dual-LLM architecture.

## Features

- **Dual-LLM Architecture**: Uses separate LLMs for generating responses (Test LLM) and evaluating them (Judge LLM)
- **Multiple Evaluation Modes**:
  - **Unified Mode**: End-to-end assessment where the system automatically generates responses using a test LLM and evaluates them using a judge LLM in a single run
  - **Two-Phase Mode**: 
    1. **Answer Generation**: Generate and persist answers from the test LLM to a structured output file (JSONL)
    2. **Judgment**: Load the pre-generated answer file and perform evaluation using the judge LLM to produce final metrics
- **Extensible Task System**: Easy to add new evaluation tasks by implementing the Task interface
- **Flexible Configuration**: Specify API endpoints, keys, and model identifiers for both Test LLM and Judge LLM independently
- **Dataset Support**: Built-in support for GSM8K dataset, with extensible architecture for additional datasets
- **Robust LLM Clients**: Support for major provider APIs (OpenAI, Anthropic, OpenAI-compatible) with error handling, retry logic, and rate limiting
- **Comprehensive Output**: Generate human-readable evaluation reports (console summary, Markdown) and structured machine-readable results (JSON) with metrics (accuracy, pass rates, confidence intervals)

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

If the `llm-eval` command is not found after installation, you can also run the tool using `python -m llm_eval.cli` (replace `llm-eval` with `python -m llm_eval.cli` in the examples below).

### Unified Mode

**Using OpenAI-compatible API (e.g., local LLM server):**

```bash
llm-eval --test-api openai-compatible --test-model Qwen2.5-7B --test-api-base http://127.0.0.1:8000/v1 --test-batch-size 8 --judge-api openai-compatible --judge-model Qwen2.5-7B --judge-api-base http://127.0.0.1:8000/v1 --judge-batch-size 8 --tasks gsm8k --output-dir ./results
```

### Two-Phase Mode

**Phase 1 (Generate answers):**

```bash
llm-eval --mode answer --test-api openai-compatible --test-model Qwen2.5-7B --test-api-key "" --test-api-base http://127.0.0.1:8000/v1  --test-batch-size 16 --tasks gsm8k
```

**Phase 2 (Judge answers):**

```bash
llm-eval --mode judge --judge-api openai-compatible --judge-model Qwen2.5-7B --judge-api-base http://127.0.0.1:8000/v1 --judge-batch-size 16 --tasks gsm8k --answers-file ./results/gsm8k/Qwen2.5-7B-gsm8k-ANSWER-20251207-143550.jsonl
```

## Adding New Tasks

To add a new evaluation task, you need to:

1. Create a new Python file in the `llm_eval/tasks/` directory (e.g., `my_task.py`)
2. Implement a subclass of the `Task` base class from `llm_eval/tasks/base.py`
3. Register your task in `llm_eval/tasks/__init__.py`

### Step-by-step Guide

#### 1. Create Task Implementation

Create a new file in `llm_eval/tasks/` (e.g., `my_task.py`) and implement all abstract methods from the `Task` base class:

- `load_samples`: Load your dataset and return a list of samples
- `build_test_prompt`: Create prompts for the test LLM
- `build_judge_prompt`: Create prompts for the judge LLM
- `parse_judgment`: Parse the judge LLM's response
- `calculate_metrics`: Calculate metrics from judgments
- `name` property: Return the task name

#### 2. Register Your Task

Add your task to the `_TASK_REGISTRY` in `llm_eval/tasks/__init__.py`:

```python
from llm_eval.tasks.my_task import MyTask

_TASK_REGISTRY = {
    "gsm8k": GS8MKTask,
    "my_task": MyTask,  # Add this line
}
```

Now you can use your task with `--tasks my_task`.

## Configuration

The tool uses command-line arguments for configuration. Key options:

- `--mode`: `unified` (default) , `answer`, or `judge`
- `--test-api`: `openai` or `openai-compatible`
- `--test-model`: model identifier (e.g., `gpt-4`, `claude-3-opus`)
- `--test-api-key`: API key (optional, can be set via environment variable)
- `--test-api-base`: Base URL for OpenAI-compatible API (optional, defaults to OpenAI's official endpoint). Should include the version path, e.g., `http://localhost:8000/v1` (not `http://localhost:8000/v1/chat/completions`). The client will append the appropriate endpoint path.
- `--test-batch-size`: Batch size for concurrent test LLM requests (default: 1). Increase to improve throughput.
- `--judge-api`, `--judge-model`, `--judge-api-key`, `--judge-api-base`: analogous options for judge LLM
- `--judge-batch-size`: Batch size for concurrent judge LLM requests (default: 1).
- `--tasks`: comma-separated list of task names (e.g., gsm8k,math).
- `--answers-file`: path to pre-generated answers for judgment
- `--output-dir`: directory for all outputs (reports, logs)
- `--seed`: random seed for reproducibility (default: 42)
- `--log-level`: logging level (DEBUG, INFO, WARNING, ERROR)

**Note for OpenAI‑compatible APIs:** The `--test-api-base` and `--judge-api-base` should be the base URL that includes the version path (e.g., `http://localhost:8000/v1`). Do not include endpoint paths like `/chat/completions` or `/completions`; the client will append them automatically. If the server requires an API key but you are using a local server without authentication, you can pass an empty string `--test-api-key ""` (the tool will treat empty strings as `None`).

## Project Structure

```
llm_eval/
├── __init__.py
├── config.py          # Configuration models and parsing
├── cli.py             # CLI entry point
├── tasks.json         # Task configurations
├── llm/
│   ├── __init__.py
│   ├── client.py      # LLM client wrappers
│   └── providers.py   # API-specific implementations
├── tasks/
│   ├── __init__.py    # Task registry
│   ├── base.py        # Task base class
│   ├── demo_task.py   # Template for new tasks
│   └── gsm8k.py       # GSM8K task implementation
└── evaluate/
    ├── __init__.py
    └── task_pipeline.py # Evaluation pipelines
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Format with black:

```bash
black llm_eval/
```

Type checking with mypy:

```bash
mypy llm_eval/
```

## License

MIT