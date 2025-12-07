"""
Task registry for LLM evaluation.
"""
from llm_eval.tasks.base import Task
from llm_eval.tasks.gsm8k import GS8MKTask

_TASK_REGISTRY = {
    "gsm8k": GS8MKTask,
}


def get_task(task_name: str) -> Task:
    """Return a task instance by name."""
    if task_name not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(_TASK_REGISTRY.keys())}")
    return _TASK_REGISTRY[task_name]()