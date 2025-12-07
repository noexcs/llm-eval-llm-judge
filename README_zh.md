# LLM 评估命令行工具

一个基于双LLM架构的生产级Python命令行界面(LLM)工具，用于全面的LLM任务评估。

## 功能特性

- **双LLM架构**：使用独立的LLM生成响应（测试LLM）和评估响应（评委LLM）
- **多种评估模式**：
  - **统一模式**：端到端评估，在一次运行中自动使用测试LLM生成响应并使用评委LLM进行评估
  - **两阶段模式**：
    1. **答案生成**：使用测试LLM生成并将答案保存到结构化输出文件(JSONL)
    2. **评判**：加载预生成的答案文件，使用评委LLM进行评估并生成最终指标
- **可扩展的任务系统**：通过实现Task接口轻松添加新的评估任务
- **灵活配置**：独立指定测试LLM和评委LLM的API端点、密钥和模型标识符
- **数据集支持**：内置GSM8K数据集支持，架构可扩展以支持更多数据集
- **强大的LLM客户端**：支持主流提供商API（OpenAI、Anthropic、OpenAI兼容）具备错误处理、重试逻辑和速率限制
- **全面输出**：生成人类可读的评估报告（控制台摘要、Markdown）和结构化的机器可读结果（JSON），包含指标（准确性、通过率、置信区间）

## 安装

```bash
pip install -e .
```

开发环境安装：

```bash
pip install -e ".[dev]"
```

## 使用方法

安装后如果找不到`llm-eval`命令，也可以使用`python -m llm_eval.cli`运行工具（在下面的示例中将`llm-eval`替换为`python -m llm_eval.cli`）。

### 统一模式

**使用OpenAI兼容API（例如本地LLM服务器）：**

```bash
llm-eval --test-api openai-compatible --test-model DeepSeek-R1-Distill-Qwen-7B --test-api-base http://127.0.0.1:8000/v1 --test-batch-size 16 --judge-api openai-compatible --judge-model DeepSeek-R1-Distill-Qwen-7B --judge-api-base http://127.0.0.1:8000/v1 --judge-batch-size 16 --tasks gsm8k --output-dir ./results
```

### 两阶段模式

**第一阶段（生成答案）：**

```bash
llm-eval --mode answer --test-api openai-compatible --test-model DeepSeek-R1-Distill-Qwen-7B --test-api-key "" --test-api-base http://127.0.0.1:8000/v1  --test-batch-size 16 --tasks gsm8k
```

**第二阶段（评判答案）：**

```bash
llm-eval --mode judge --judge-api openai-compatible --judge-model DeepSeek-R1-Distill-Qwen-7B --judge-api-base http://127.0.0.1:8000/v1 --judge-batch-size 16 --answers-file ./generated_answers.jsonl
```

## 添加新任务

要添加新的评估任务，需要：

1. 在`llm_eval/tasks/`目录中创建新的Python文件（例如`my_task.py`）
2. 实现来自`llm_eval/tasks/base.py`的`Task`基类的子类
3. 在`llm_eval/tasks/__init__.py`中注册你的任务

### 分步指南

#### 1. 创建任务实现

在`llm_eval/tasks/`中创建新文件（例如`my_task.py`）并实现`Task`基类的所有抽象方法：

- `load_samples`：加载数据集并返回样本列表
- `build_test_prompt`：为测试LLM创建提示
- `build_judge_prompt`：为评委LLM创建提示
- `parse_judgment`：解析评委LLM的响应
- `calculate_metrics`：从评判中计算指标
- `name`属性：返回任务名称

#### 2. 注册你的任务

将你的任务添加到`llm_eval/tasks/__init__.py`中的`_TASK_REGISTRY`：

```python
from llm_eval.tasks.my_task import MyTask

_TASK_REGISTRY = {
    "gsm8k": GS8MKTask,
    "my_task": MyTask,  # 添加这一行
}
```

现在你可以使用`--tasks my_task`来使用你的任务。

## 配置

工具使用命令行参数进行配置。主要选项：

- `--mode`：`unified`（默认）、`answer`或`judge`
- `--test-api`：`openai`或`openai-compatible`
- `--test-model`：模型标识符（例如`gpt-4`、`claude-3-opus`）
- `--test-api-key`：API密钥（可选，可通过环境变量设置）
- `--test-api-base`：OpenAI兼容API的基础URL（可选，默认为OpenAI官方端点）。应包含版本路径，例如`http://localhost:8000/v1`（而不是`http://localhost:8000/v1/chat/completions`）。客户端会自动附加相应的端点路径。
- `--test-batch-size`：并发测试LLM请求的批处理大小（默认：1）。增加以提高吞吐量。
- `--judge-api`、`--judge-model`、`--judge-api-key`、`--judge-api-base`：评委LLM的类似选项
- `--judge-batch-size`：并发评委LLM请求的批处理大小（默认：1）。
- `--tasks`：逗号分隔的任务名称列表（例如gsm8k,math）。
- `--answers-file`：预生成答案的路径，用于评判
- `--output-dir`：所有输出的目录（报告、日志）
- `--seed`：随机种子以确保可重现性（默认：42）
- `--log-level`：日志级别（DEBUG、INFO、WARNING、ERROR）

**OpenAI兼容API注意事项：** `--test-api-base`和`--judge-api-base`应该是包含版本路径的基础URL（例如`http://localhost:8000/v1`）。不要包含端点路径如`/chat/completions`或`/completions`；客户端会自动附加它们。如果服务器需要API密钥但你使用的是没有身份验证的本地服务器，可以传递空字符串`--test-api-key ""`（工具会将空字符串视为`None`）。

## 项目结构

```
llm_eval/
├── __init__.py
├── config.py          # 配置模型和解析
├── cli.py             # CLI入口点
├── tasks.json         # 任务配置
├── llm/
│   ├── __init__.py
│   ├── client.py      # LLM客户端包装器
│   └── providers.py   # API特定实现
├── tasks/
│   ├── __init__.py    # 任务注册表
│   ├── base.py        # 任务基类
│   ├── demo_task.py   # 新任务模板
│   └── gsm8k.py       # GSM8K任务实现
└── evaluate/
    ├── __init__.py
    └── task_pipeline.py # 评估流水线
```

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码风格

使用black格式化：

```bash
black llm_eval/
```

使用mypy进行类型检查：

```bash
mypy llm_eval/
```

## 许可证

MIT