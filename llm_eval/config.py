"""
Configuration models for LLM evaluation.
"""
from enum import Enum
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai-compatible"


class EvaluationMode(str, Enum):
    """Evaluation modes."""
    UNIFIED = "unified"
    ANSWER = "answer"
    JUDGE = "judge"


class LLMConfig(BaseModel):
    """Configuration for an LLM."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = Field(default=None, description="API key (can be set via env var)")
    api_base: Optional[str] = Field(default=None, description="Base URL for API (for OpenAI-compatible)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    batch_size: int = Field(default=1, gt=0, description="Batch size for concurrent requests")

    @validator("api_key", "api_base")
    def empty_to_none(cls, v):
        """Convert empty strings to None."""
        if v == "":
            return None
        return v

    class Config:
        extra = "forbid"


class OutputConfig(BaseModel):
    """Configuration for output."""
    output_dir: Path = Field(default=Path("./results"), description="Directory for all outputs")
    answers_file: Optional[Path] = Field(default=None, description="Path to pre-generated answers for judgment")

    @validator("output_dir")
    def create_output_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""
    mode: EvaluationMode = Field(default=EvaluationMode.UNIFIED, description="Evaluation mode")
    task_name: str = Field(default=None, description="Task name")
    test_llm: Optional[LLMConfig] = Field(default=None, description="Configuration for test LLM")
    judge_llm: Optional[LLMConfig] = Field(default=None, description="Configuration for judge LLM")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        extra = "forbid"
