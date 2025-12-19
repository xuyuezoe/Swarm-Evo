"""Prompt 管理模块，负责拼装动态补充信息。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
from jinja2 import Environment, FileSystemLoader

@dataclass
class PromptContext:
    """提示上下文数据结构，描述动态变量与运行阶段。"""

    # Common fields
    workspace_root: str
    conda_env_name: str
    time_limit_seconds: int
    total_iterations: int
    iteration: int
    elapsed_seconds: float
    remaining_seconds: float
    conda_packages: str
    task_description: str  # Updated from task_goal as per feedback

    # Control fields
    template_name: Optional[str] = None

    # Explore fields
    parent_code: Optional[str] = None
    parent_feedback: Optional[str] = None
    parent_history: Optional[str] = None

    # Merge/Select fields
    candidates: Optional[Dict[str, str]] = None
    gene_plan: Optional[Dict[str, Any]] = None

    # Evaluate fields
    solution_code: Optional[str] = None
    execution_logs: Optional[str] = None

    # Memory injections
    global_prior: str = ""
    local_report: str = ""


class PromptManager:
    """提示词管理器，负责拼接动态补充信息与模板注入。"""

    def __init__(self, template_dir: str = "benchmark/mle-bench/prompt_templates") -> None:
        """初始化提示词管理器。

        参数:
            template_dir: 外置模板目录，默认使用 benchmark/mle-bench/prompt_templates。
        """
        self._template_dir = Path(template_dir)
        self._env = Environment(loader=FileSystemLoader(self._template_dir))

    def build_system_prompt(self, context: Optional[PromptContext] = None) -> str:
        """构建 System Prompt。
        
        注意：目前 system_prompt.j2 不接受任何动态变量。
        """
        template = self._env.get_template("system_prompt.j2")
        return template.render()

    def build_user_prompt(self, context: PromptContext, history: str) -> str:
        """构建 User Prompt。

        参数:
            context: 运行时动态上下文数据。
            history: ReAct 执行历史。
        返回:
            拼接完模板与补充信息后的完整用户提示字符串。
        """
        if not context.template_name:
            template_name = "explore_user_prompt.j2" 
        else:
            template_name = context.template_name

        template = self._env.get_template(template_name)
        
        # 准备环境信息
        elapsed = self._format_duration(context.elapsed_seconds)
        remaining = self._format_duration(context.remaining_seconds)
        total_time = self._format_duration(context.time_limit_seconds)
        
        # 准备历史信息
        history_block = self._build_history_block(history)

        return template.render(
            # Common
            workspace_root=context.workspace_root,
            conda_env_name=context.conda_env_name,
            time_limit=self._format_hours(context.time_limit_seconds),
            
            task_description=context.task_description,
            elapsed=elapsed,
            remaining=remaining,
            total_time=total_time,
            iteration=context.iteration,
            total_iterations=context.total_iterations,
            conda_packages=context.conda_packages.strip(),
            history_block=history_block,

            # Explore
            parent_code=context.parent_code,
            parent_feedback=context.parent_feedback,
            parent_history=context.parent_history,

            # Merge / Select
            candidates=context.candidates,
            gene_plan=context.gene_plan,

            # Evaluate
            solution_code=context.solution_code,
            execution_logs=context.execution_logs,
            global_prior=context.global_prior,
            local_report=context.local_report
        )

    def _build_history_block(self, history: str) -> str:
        """
        生成历史记录区块，若无历史则返回空字符串。
        """
        if not history or not history.strip():
            return "No history available."
        return history

    def _format_duration(self, seconds: float) -> str:
        """将秒数转换为“分秒”文本。"""
        safe_seconds = max(0, int(seconds))
        minutes, remain = divmod(safe_seconds, 60)
        return f"{minutes}分{remain}秒"

    def _format_hours(self, seconds: float) -> str:
        """将秒数转换为小时（保留1位小数）。"""
        return f"{seconds / 3600:.1f}"
