"""
Experiment Framework for BingFu paper evaluation.

Provides:
  1. Benchmark task definitions (20 tasks x 5 categories)
  2. Baseline wrappers (GPT-4 Solo, AutoGen-style, CrewAI-style, MetaGPT-style)
  3. Evaluation metrics (SR, TCS, Efficiency, Utilization, TTA, EQ)
  4. Experiment runner with statistical analysis
  5. Result persistence and reporting

Usage:
    from bingfu.experiment import ExperimentRunner, BENCHMARK_TASKS
    runner = ExperimentRunner(llm_provider=my_llm)
    results = runner.run_all(seeds=[42, 123, 456])
    runner.print_report(results)
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
import re
import time
import os
import statistics
import math
import numpy as np

from .agent import Agent
from .commander import Commander
from .tactics import TacticEngine
from .tactic_library import get_tactic_library
from .profile import GeneralProfile, CombatStyle, CombatStats
from .presets import get_preset, PRESET_GENERALS
from .assessment import TaskAssessor, TaskAssessment
from .matcher import TaskMatcher, MatchResult
from .harness import call_structured, AgentHarnessFailure, HarnessResult
from .verify import verify_output, VerifyResult
from .graph import GraphOrchestrator, route_for, RoutePlan
from .tools import belt_for, ToolBelt


# ═══════════════════════════════════════════════════════════════
# Benchmark Tasks
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkTask:
    """A single benchmark task with ground truth evaluation criteria."""
    id: str
    category: str  # IR, Code, Data, Write, Reason
    description: str
    complexity: int  # 1-10
    required_capabilities: List[str]
    evaluation_criteria: List[str]
    expected_output_type: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "complexity": self.complexity,
            "required_capabilities": self.required_capabilities,
            "evaluation_criteria": self.evaluation_criteria,
            "expected_output_type": self.expected_output_type,
        }


BENCHMARK_TASKS: List[BenchmarkTask] = [
    # ── Information Retrieval (IR) ──
    BenchmarkTask(
        id="IR1", category="IR", complexity=4,
        description="Research and summarize the current state of quantum computing hardware, "
                    "including major companies, qubit technologies, and recent breakthroughs. "
                    "Provide a structured report with key findings and citations.",
        required_capabilities=["信息收集", "文案写作", "数据分析"],
        evaluation_criteria=["Factual Accuracy", "Coverage", "Structure", "Citation Quality"],
        expected_output_type="Research Report",
    ),
    BenchmarkTask(
        id="IR2", category="IR", complexity=3,
        description="Find and compare three open-source LLM frameworks (excluding LangChain). "
                    "Compare their features, performance benchmarks, community size, and use cases. "
                    "Present findings in a comparison table with recommendations.",
        required_capabilities=["信息收集", "数据分析"],
        evaluation_criteria=["Completeness", "Comparison Depth", "Recommendation Quality"],
        expected_output_type="Comparison Report",
    ),
    BenchmarkTask(
        id="IR3", category="IR", complexity=5,
        description="Investigate the environmental impact of training large AI models. "
                    "Gather data on carbon emissions, water usage, and energy consumption from "
                    "recent studies. Create a fact-based summary with quantitative evidence.",
        required_capabilities=["信息收集", "数据分析", "文案写作"],
        evaluation_criteria=["Factual Accuracy", "Quantitative Evidence", "Source Diversity"],
        expected_output_type="Fact-based Summary",
    ),
    BenchmarkTask(
        id="IR4", category="IR", complexity=6,
        description="Research the regulatory landscape for AI in the EU, US, and China. "
                    "Compare the EU AI Act, US Executive Orders, and China's AI regulations. "
                    "Identify key differences and implications for AI developers.",
        required_capabilities=["信息收集", "战略规划", "文案写作"],
        evaluation_criteria=["Regulatory Accuracy", "Comparative Analysis", "Practical Implications"],
        expected_output_type="Policy Analysis",
    ),

    # ── Code Generation (Code) ──
    BenchmarkTask(
        id="C1", category="Code", complexity=5,
        description="Implement a RESTful API server in Python using FastAPI that provides "
                    "CRUD operations for a 'tasks' resource. Include input validation, "
                    "error handling, and async database operations with SQLAlchemy.",
        required_capabilities=["代码开发", "系统架构"],
        evaluation_criteria=["Correctness", "Code Quality", "Error Handling", "Completeness"],
        expected_output_type="Python Code",
    ),
    BenchmarkTask(
        id="C2", category="Code", complexity=7,
        description="Build a full-stack web application: a simple project management dashboard "
                    "with user authentication (JWT), task CRUD, real-time updates via WebSocket, "
                    "and a responsive UI. Backend: FastAPI. Frontend: plain HTML/JS or React.",
        required_capabilities=["代码开发", "系统架构", "设计创意"],
        evaluation_criteria=["Functionality", "Code Structure", "Security", "UI Quality"],
        expected_output_type="Full-Stack Application",
    ),
    BenchmarkTask(
        id="C3", category="Code", complexity=6,
        description="Write a data processing pipeline in Python that: (1) reads CSV files "
                    "from a directory, (2) cleans and normalizes data, (3) performs statistical "
                    "analysis, (4) generates visualizations, (5) exports results to JSON. "
                    "Include proper logging and error handling.",
        required_capabilities=["代码开发", "数据分析"],
        evaluation_criteria=["Correctness", "Pipeline Design", "Error Handling", "Output Quality"],
        expected_output_type="Python Script",
    ),
    BenchmarkTask(
        id="C4", category="Code", complexity=4,
        description="Implement a rate-limited API client with retry logic, exponential backoff, "
                    "and request caching. The client should handle pagination, respect rate limits, "
                    "and provide progress reporting. Write in Python with type hints.",
        required_capabilities=["代码开发"],
        evaluation_criteria=["Correctness", "Error Handling", "Code Quality", "Documentation"],
        expected_output_type="Python Library",
    ),

    # ── Data Analysis (Data) ──
    BenchmarkTask(
        id="D1", category="Data", complexity=5,
        description="Analyze a sales dataset (provided as CSV with columns: date, product, "
                    "region, quantity, revenue). Perform trend analysis, identify top products "
                    "and regions, detect seasonal patterns, and generate visualizations. "
                    "Write a comprehensive analysis report.",
        required_capabilities=["数据分析", "文案写作"],
        evaluation_criteria=["Analysis Depth", "Visualization Quality", "Insight Quality"],
        expected_output_type="Analysis Report + Visualizations",
    ),
    BenchmarkTask(
        id="D2", category="Data", complexity=7,
        description="Perform multi-dimensional customer segmentation analysis: combine "
                    "demographic, behavioral, and transactional data. Use clustering techniques, "
                    "profile each segment, and provide actionable marketing recommendations. "
                    "Include statistical validation of segment differences.",
        required_capabilities=["数据分析", "战略规划", "代码开发"],
        evaluation_criteria=["Segmentation Quality", "Statistical Rigor", "Recommendation Quality"],
        expected_output_type="Segmentation Analysis",
    ),
    BenchmarkTask(
        id="D3", category="Data", complexity=6,
        description="Conduct an A/B test analysis: given experimental data with control and "
                    "treatment groups, perform statistical testing (t-test, chi-square), "
                    "calculate effect sizes and confidence intervals, check for SRM, "
                    "and produce a formal experiment report with conclusions.",
        required_capabilities=["数据分析", "战略规划"],
        evaluation_criteria=["Statistical Correctness", "Analysis Completeness", "Report Quality"],
        expected_output_type="Experiment Report",
    ),
    BenchmarkTask(
        id="D4", category="Data", complexity=4,
        description="Create an interactive data dashboard specification: define KPIs, "
                    "design chart layouts, specify data transformations, and write the "
                    "dashboard configuration. Include mock data to demonstrate the design.",
        required_capabilities=["数据分析", "设计创意", "文案写作"],
        evaluation_criteria=["KPI Selection", "Design Quality", "Specification Completeness"],
        expected_output_type="Dashboard Specification",
    ),

    # ── Content Creation (Write) ──
    BenchmarkTask(
        id="W1", category="Write", complexity=5,
        description="Write a comprehensive technical blog post (2000-3000 words) explaining "
                    "how transformer models work. Target audience: software engineers with "
                    "basic ML knowledge. Include diagrams (described in text), code snippets, "
                    "and intuitive explanations of attention mechanisms.",
        required_capabilities=["文案写作", "沟通协调"],
        evaluation_criteria=["Clarity", "Technical Accuracy", "Engagement", "Structure"],
        expected_output_type="Technical Blog Post",
    ),
    BenchmarkTask(
        id="W2", category="Write", complexity=6,
        description="Create a multi-channel content strategy for a new AI developer tool launch: "
                    "Twitter/X thread (10 tweets), LinkedIn article (800 words), "
                    "product launch blog post (1500 words), and email newsletter. "
                    "All content should be cohesive and tailored to each platform.",
        required_capabilities=["文案写作", "设计创意", "沟通协调"],
        evaluation_criteria=["Platform Appropriateness", "Message Consistency", "Creativity", "Call-to-Action"],
        expected_output_type="Multi-Channel Content",
    ),
    BenchmarkTask(
        id="W3", category="Write", complexity=4,
        description="Write a product requirements document (PRD) for a mobile fitness tracking app. "
                    "Include: problem statement, user personas, feature specifications, "
                    "user stories, success metrics, and a phased rollout plan.",
        required_capabilities=["文案写作", "战略规划"],
        evaluation_criteria=["Completeness", "Clarity", "Feasibility", "Structure"],
        expected_output_type="PRD Document",
    ),
    BenchmarkTask(
        id="W4", category="Write", complexity=3,
        description="Write a clear, empathetic response to a customer complaint about a "
                    "billing error where they were overcharged $50 for three consecutive months. "
                    "Acknowledge the issue, explain the resolution, offer compensation, "
                    "and rebuild trust. Tone: professional, empathetic, solution-oriented.",
        required_capabilities=["文案写作", "沟通协调"],
        evaluation_criteria=["Empathy", "Clarity", "Resolution Quality", "Tone"],
        expected_output_type="Customer Response",
    ),

    # ── Logical Reasoning (Reason) ──
    BenchmarkTask(
        id="R1", category="Reason", complexity=8,
        description="A company must decide between three strategic options: (A) expand into "
                    "a new market with high growth but regulatory uncertainty, (B) deepen "
                    "penetration in existing markets with lower growth but known risks, or "
                    "(C) acquire a competitor to gain market share quickly. Analyze each option "
                    "using a decision matrix, consider second-order effects, and recommend "
                    "a strategy with clear reasoning.",
        required_capabilities=["战略规划", "数据分析"],
        evaluation_criteria=["Analysis Depth", "Logical Consistency", "Recommendation Quality"],
        expected_output_type="Strategic Analysis",
    ),
    BenchmarkTask(
        id="R2", category="Reason", complexity=7,
        description="Solve the following puzzle: 'There are five houses in a row, each of a "
                    "different color and inhabited by people of different nationalities, with "
                    "different pets, drinks, and cigarettes. Given 15 clues (similar to Einstein's "
                    "riddle), determine who owns the fish.' Systematically work through each clue, "
                    "build a constraint table, and verify your solution.",
        required_capabilities=["战略规划"],
        evaluation_criteria=["Correctness", "Solution Method", "Clarity of Explanation"],
        expected_output_type="Logic Puzzle Solution",
    ),
    BenchmarkTask(
        id="R3", category="Reason", complexity=6,
        description="Evaluate the ethical implications of deploying an AI hiring system that "
                    "has shown 5% higher accuracy than human recruiters but exhibits a 3% "
                    "demographic bias. Apply multiple ethical frameworks (utilitarian, "
                    "deontological, virtue ethics) and recommend a deployment decision "
                    "with justification.",
        required_capabilities=["战略规划", "文案写作"],
        evaluation_criteria=["Ethical Reasoning", "Framework Application", "Recommendation Balance"],
        expected_output_type="Ethical Analysis",
    ),
    BenchmarkTask(
        id="R4", category="Reason", complexity=5,
        description="A startup has $2M in funding, 12 months of runway, and needs to choose "
                    "between: (A) building a full product before launch (6 months dev, then "
                    "launch), or (B) launching an MVP in 2 months and iterating. Analyze both "
                    "approaches considering market timing, competitive pressure, team capability, "
                    "and funding requirements. Provide a quantitative decision model.",
        required_capabilities=["战略规划", "数据分析"],
        evaluation_criteria=["Analysis Depth", "Quantitative Rigor", "Practicality"],
        expected_output_type="Decision Analysis",
    ),
]


# ═══════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    method: str
    seed: int
    success: bool
    completion_score: float  # 1-5 TCS
    tokens_consumed: int
    agent_utilization: float  # 0-1
    tactic_used: str = ""
    tactic_alignment: float = 0.0  # human eval placeholder
    execution_time: float = 0.0
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ★ 验收轨迹必须进结果记录，否则 VERIFY 层不可测量。
    #
    #   只看成功率的话，「一次过」和「回炉两轮才修好」完全无法区分 ——
    #   于是"加了门禁之后 Code 类变好了"这句话没有证据支撑：
    #   它可能只是模型当天状态好。
    #
    #   有了轨迹才能回答：门禁触发了几次？触发后有多少真的被修好了？
    #   代价是多少 token？没有这三个数，这一层就是又一个
    #   "看起来很负责但无法证伪"的东西。
    verify_revisions: int = 0          # 回炉了几轮（0 = 一次过）
    verify_passed: bool = True         # 最终是否通过验收
    verify_stopped_by: str = ""        # 若未通过，是被哪条边界叫停的
    verify_checks: List[str] = field(default_factory=list)

    # ★ 工具调用统计必须进结果记录。
    #   "挂了工具但模型从不调用"是完全静默的失效 ——
    #   成功率不变、日志无异常，只有这个数能暴露它。
    tool_calls: Dict[str, int] = field(default_factory=dict)
    output_len: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "seed": self.seed,
            "success": self.success,
            "completion_score": self.completion_score,
            "tokens_consumed": self.tokens_consumed,
            "agent_utilization": self.agent_utilization,
            "tactic_used": self.tactic_used,
            "tactic_alignment": self.tactic_alignment,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
            "verify_revisions": self.verify_revisions,
            "verify_passed": self.verify_passed,
            "verify_stopped_by": self.verify_stopped_by,
            "verify_checks": list(self.verify_checks),
            "tool_calls": dict(self.tool_calls),
            "output_len": self.output_len,
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics across tasks for one method."""
    method: str
    avg_sr: float
    std_sr: float
    avg_tcs: float
    std_tcs: float
    avg_tokens: float
    avg_utilization: float
    per_category_sr: Dict[str, float] = field(default_factory=dict)
    per_task_results: List[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "avg_sr": self.avg_sr,
            "std_sr": self.std_sr,
            "avg_tcs": self.avg_tcs,
            "std_tcs": self.std_tcs,
            "avg_tokens": self.avg_tokens,
            "avg_utilization": self.avg_utilization,
            "per_category_sr": self.per_category_sr,
        }


# ═══════════════════════════════════════════════════════════════
# LLM-as-Judge Evaluator
# ═══════════════════════════════════════════════════════════════

class LLMJudge:
    """
    LLM-as-Judge for automated task evaluation.

    Evaluates task outputs on success (binary), completion score (1-5),
    and provides structured feedback. Used as primary evaluation method
    with human spot-check for calibration.
    """

    EVAL_PROMPT = """You are an expert evaluator for AI agent task outputs.
Evaluate the following task output against the given criteria.

Task: {task_description}
Task Category: {category}
Expected Output Type: {expected_type}
Evaluation Criteria: {criteria}

Agent Output:
---
{output}
---

Please evaluate and respond in JSON format:
{{
    "success": true/false,  // Was the task completed successfully?
    "completion_score": 1-5,  // 1=completely failed, 3=partially complete, 5=excellent
    "criteria_scores": {{"criterion_name": score}},  // Score each criterion 1-5
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "overall_assessment": "brief overall assessment"
}}

Only respond with the JSON. No other text."""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def evaluate(
        self,
        task: BenchmarkTask,
        output: str,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate a task output.

        Returns:
            (success, completion_score, detailed_metrics)
        """
        if self.llm:
            return self._evaluate_with_llm(task, output)
        return self._evaluate_with_heuristics(task, output)

    def _evaluate_with_llm(
        self, task: BenchmarkTask, output: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """LLM-based evaluation."""
        from .llm.base import LLMMessage, RoleType
        prompt = self.EVAL_PROMPT.format(
            task_description=task.description,
            category=task.category,
            expected_type=task.expected_output_type,
            criteria=", ".join(task.evaluation_criteria),
            output=output[:8000],  # Truncate for token limits
        )
        try:
            messages = [
                LLMMessage(role=RoleType.SYSTEM,
                          content="You are an expert evaluator. Respond only with valid JSON."),
                LLMMessage(role=RoleType.USER, content=prompt),
            ]
            # ★ 走 Harness 的五级降级链，不再自己解析。
            #
            #   这里原先是「一次调用 + 一行正则」。正则取错了嵌套层，
            #   于是整套评估在结构上不可能报告成功 ——
            #   而且**没有任何症状**。
            #
            #   现在：Level 0/1 修不好就降温重生成（Level 2），
            #   再不行走模板并标 degraded，最后耗尽则抛异常。
            #   关键是每一级都被记下来，而不是悄悄变成一个合法数字。
            def _gen(temperature: float = 0.1) -> str:
                resp = _generate_checked(
                    self.llm, messages, temperature=temperature, max_tokens=1024
                )
                return resp.content or ""

            harness_result = call_structured(
                _gen,
                required=("success", "completion_score"),
                template=None,          # 裁判**不给模板兜底**：
                                        # 编一个分数比没有分数更糟
                max_regenerate=1,
            )
            data = harness_result.output

            # ★★ 这里原先是：
            #        re.search(r'\{[^{}]*\}', content, re.DOTALL)
            #
            #    `[^{}]*` = 「不含花括号的字符」，所以它匹配的是**最内层**的
            #    花括号对 —— 也就是模型回复里嵌套的 "criteria_scores": {...}，
            #    而不是外层那个含 success / completion_score 的信封。
            #
            #    实测模型返回的是完全合规的 JSON，而解析器拿到的是：
            #        {"Factual Accuracy": 1, "Coverage": 1, "Structure": 2, ...}
            #    于是 data.get("success", False) 恒为 False、
            #    data.get("completion_score", 3.0) 恒为 3.0。
            #
            #    ★ 后果：**这套评估在结构上不可能报告成功**。
            #      七个方法全线 0.0% SR、TCS 全是 3.00，与模型好坏、
            #      与网络通不通都无关 —— 而它看起来完全像一组正常的实验结果。
            #
            #    提示词是对的，模型答得也对，只有这一行正则错了；
            #    而它错得**没有任何症状**：不报错、不抛异常、返回合法数字。
            success = bool(data.get("success", False))
            score = float(data.get("completion_score", 3.0))
            # ★ 把降级等级带回去 —— 一个 level=2 重生成得到的分数
            #   和一个 level=0 一次过的分数，在统计上不该被同等对待。
            data = dict(data)
            data["_harness"] = harness_result.to_dict()
            return success, min(5.0, max(1.0, score)), data

        except AgentHarnessFailure:
            # ★ 裁判五级耗尽 = 这一条**没测成**，不是"方法没做到"。
            #   与 LLMCallFailed 同类：往上抛，由 run_* 记成基础设施故障。
            #   降级到启发式（按输出长度打分）会让报告上出现一整套
            #   看似正常、实则与论文所述方法无关的数字。
            raise
        except LLMCallFailed:
            # ★ 基础设施故障**不降级** —— 往上抛。
            #
            #   这里原先一个 `except Exception` 把所有情况都退回
            #   `_evaluate_with_heuristics`（按输出长度打分）。
            #   于是 LLM 裁判根本没跑成的时候，报告上照样有
            #   一整套成功率数字，而**没有任何地方说明裁判失效了**。
            #
            #   论文声称的是 LLM-as-judge。悄悄换成"看输出有没有超过 500 字"，
            #   得到的数字与论文描述的方法不是同一件事。
            raise
        except Exception:
            # ★ 模型回了但格式不对（JSON 解析失败）——这属于**可恢复**，
            #   降级到启发式是合理的。与上面那条的分界是：
            #   连不上是环境的问题，回复难看是模型的问题。
            return self._evaluate_with_heuristics(task, output)

    def _evaluate_with_heuristics(
        self, task: BenchmarkTask, output: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Heuristic fallback evaluation (no LLM available)."""
        # Basic heuristics
        output_len = len(output)
        has_structure = any(marker in output for marker in ['#', '##', '1.', '- ', '**'])
        has_code = '```' in output or 'def ' in output or 'import ' in output

        # Score based on output characteristics
        if output_len < 50:
            score = 1.0
            success = False
        elif output_len < 200:
            score = 2.0
            success = False
        elif output_len < 500:
            score = 3.0
            success = has_structure
        elif output_len < 2000:
            score = 4.0 if has_structure else 3.0
            success = has_structure
        else:
            score = 5.0 if (has_structure and (has_code or task.category != "Code")) else 4.0
            success = True

        if task.category == "Code" and not has_code:
            score = min(score, 2.0)
            success = False

        return success, score, {
            "heuristic_only": True,
            "output_length": output_len,
            "has_structure": has_structure,
        }


# ═══════════════════════════════════════════════════════════════
# Baseline Implementations
# ═══════════════════════════════════════════════════════════════

class BaselineRunner:
    """Wrapper for baseline method execution."""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def run_gpt4_solo(self, task: BenchmarkTask, seed: int = 42) -> TaskResult:
        """Single GPT-4 call (no collaboration)."""
        t0 = time.time()
        tokens = 0
        try:
            if self.llm:
                from .llm.base import LLMMessage, RoleType
                messages = [
                    LLMMessage(role=RoleType.SYSTEM,
                              content="You are a helpful AI assistant. Complete the task thoroughly."),
                    LLMMessage(role=RoleType.USER, content=task.description),
                ]
                response = _generate_checked(self.llm, messages, temperature=0.3)
                output = response.content or ""
                tokens = response.usage.get("total_tokens", len(output)//4) if hasattr(response, 'usage') else len(output)//4
            else:
                output = f"[GPT-4 Solo Placeholder for {task.id}]"
                tokens = 500
        except Exception as e:
            output = ""
            error = str(e)
            return TaskResult(
                task_id=task.id, method="GPT-4 Solo", seed=seed,
                success=False, completion_score=1.0, tokens_consumed=tokens,
                agent_utilization=1.0, execution_time=time.time()-t0, error=error,
            )

        judge = LLMJudge(self.llm)
        success, score, _ = judge.evaluate(task, output)

        return TaskResult(
            task_id=task.id, method="GPT-4 Solo", seed=seed,
            success=success, completion_score=score, tokens_consumed=tokens,
            agent_utilization=1.0, execution_time=time.time()-t0, output=output[:500],
        )

    def run_autogen_style(self, task: BenchmarkTask, seed: int = 42) -> TaskResult:
        """
        Simulate AutoGen-style conversational collaboration.
        Two agents: one analyst + one executor, conversing to solve.
        """
        t0 = time.time()
        tokens = 0
        try:
            agent1 = Agent(name="Analyst", role="分析者",
                          description="Analyze tasks and propose solutions")
            agent2 = Agent(name="Executor", role="执行者",
                          description="Execute the proposed solution")

            if self.llm:
                # Simulate conversation: Analyst → Executor
                analyst_prompt = f"Analyze this task and propose an approach:\n\n{task.description}"
                from .llm.base import LLMMessage, RoleType
                messages1 = [
                    LLMMessage(role=RoleType.SYSTEM,
                              content="You are an analyst. Analyze the task and propose a detailed execution plan."),
                    LLMMessage(role=RoleType.USER, content=analyst_prompt),
                ]
                resp1 = _generate_checked(self.llm, messages1, temperature=0.3)
                analysis = resp1.content or ""

                executor_prompt = f"Execute this plan for the task:\n\nTask: {task.description}\n\nPlan:\n{analysis}"
                messages2 = [
                    LLMMessage(role=RoleType.SYSTEM,
                              content="You are an executor. Follow the plan and produce the final output."),
                    LLMMessage(role=RoleType.USER, content=executor_prompt),
                ]
                resp2 = _generate_checked(self.llm, messages2, temperature=0.3)
                output = resp2.content or ""

                tokens = (getattr(resp1, 'usage', {}).get('total_tokens', len(analysis)//4) +
                         getattr(resp2, 'usage', {}).get('total_tokens', len(output)//4))
            else:
                output = f"[AutoGen Placeholder for {task.id}]"
                tokens = 1000
        except Exception as e:
            return TaskResult(
                task_id=task.id, method="AutoGen", seed=seed,
                success=False, completion_score=1.0, tokens_consumed=500,
                agent_utilization=0.5, execution_time=time.time()-t0, error=str(e),
            )

        judge = LLMJudge(self.llm)
        success, score, _ = judge.evaluate(task, output)

        return TaskResult(
            task_id=task.id, method="AutoGen", seed=seed,
            success=success, completion_score=score, tokens_consumed=tokens,
            agent_utilization=0.67, execution_time=time.time()-t0, output=output[:500],
        )

    def run_crewai_style(self, task: BenchmarkTask, seed: int = 42) -> TaskResult:
        """
        Simulate CrewAI-style role-based execution.
        Single agent with predefined role matching.
        """
        t0 = time.time()
        tokens = 0
        try:
            # Simple role matching based on category
            role_map = {
                "IR": "Research Analyst",
                "Code": "Software Engineer",
                "Data": "Data Scientist",
                "Write": "Content Writer",
                "Reason": "Strategy Consultant",
            }
            role = role_map.get(task.category, "General Assistant")

            if self.llm:
                from .llm.base import LLMMessage, RoleType
                messages = [
                    LLMMessage(role=RoleType.SYSTEM,
                              content=f"You are a {role}. Complete the task using your expertise."),
                    LLMMessage(role=RoleType.USER, content=task.description),
                ]
                response = _generate_checked(self.llm, messages, temperature=0.3)
                output = response.content or ""
                tokens = getattr(response, 'usage', {}).get('total_tokens', len(output)//4)
            else:
                output = f"[CrewAI Placeholder for {task.id}]"
                tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来
        except Exception as e:
            return TaskResult(
                task_id=task.id, method="CrewAI", seed=seed,
                success=False, completion_score=1.0, tokens_consumed=500,
                agent_utilization=0.5, execution_time=time.time()-t0, error=str(e),
            )

        judge = LLMJudge(self.llm)
        success, score, _ = judge.evaluate(task, output)

        return TaskResult(
            task_id=task.id, method="CrewAI", seed=seed,
            success=success, completion_score=score, tokens_consumed=tokens,
            agent_utilization=0.5, execution_time=time.time()-t0, output=output[:500],
        )

    def run_metagpt_style(self, task: BenchmarkTask, seed: int = 42) -> TaskResult:
        """
        Simulate MetaGPT-style SOP-driven execution.
        Analyst → Architect → Developer → Reviewer pipeline.
        """
        t0 = time.time()
        tokens = 0
        try:
            if self.llm:
                from .llm.base import LLMMessage, RoleType

                # SOP phases
                phases = [
                    ("Product Manager", "Analyze requirements and create a specification"),
                    ("Architect", "Design the solution architecture"),
                    ("Developer", "Implement the solution"),
                    ("QA Engineer", "Review and validate the output"),
                ]

                context = ""
                for role_name, role_instruction in phases:
                    messages = [
                        LLMMessage(role=RoleType.SYSTEM,
                                  content=f"You are a {role_name}. {role_instruction}."),
                        LLMMessage(role=RoleType.USER,
                                  content=f"Task: {task.description}\n\nContext from previous phase:\n{context[-2000:]}"),
                    ]
                    resp = _generate_checked(self.llm, messages, temperature=0.3)
                    context += f"\n\n[{role_name} Output]:\n{resp.content or ''}"
                    if hasattr(resp, 'usage'):
                        tokens += resp.usage.get('total_tokens', 500)

                # Final output is the last phase
                output = context
            else:
                output = f"[MetaGPT Placeholder for {task.id}]"
                tokens = 2000
        except Exception as e:
            return TaskResult(
                task_id=task.id, method="MetaGPT", seed=seed,
                success=False, completion_score=1.0, tokens_consumed=1000,
                agent_utilization=0.75, execution_time=time.time()-t0, error=str(e),
            )

        judge = LLMJudge(self.llm)
        success, score, _ = judge.evaluate(task, output)

        return TaskResult(
            task_id=task.id, method="MetaGPT", seed=seed,
            success=success, completion_score=score, tokens_consumed=tokens,
            agent_utilization=0.75, execution_time=time.time()-t0, output=output[:500],
        )

    @staticmethod
    def _run_with_tactic(agent, task_description: str, result) -> Tuple[str, int]:
        """把选中的战术注入 system_prompt 后执行，返回 (输出, token 用量)。

        ════════════════════════════════════════════════════════
         ★ 抽出来是为了让消融**每档只改一个变量**
        ════════════════════════════════════════════════════════

        原设计里 `full` 相对两个消融档同时改了两件事：

            ① 选将领的方式（按战力 vs 随机）
            ② 是否把战术注入提示词

        而 `no_power` 算出的战术**从不注入**（只存进 tactic_name 用于报表），
        于是它与 `no_tactic` 在操作上完全相同 ——
        实测 29% vs 28%，p = 1.0。

        ★ 后果不是"某一档没做好"，是**整套消融无法归因**：
          full 领先的 32 个点（p<0.00001）究竟来自战术层、将领选择、
          还是提示词注入，这个设计答不出来。
          而 32 点、p<0.00001 的结果看起来极有说服力 ——
          这正是它危险的地方。
        """

        original = agent.system_prompt
        agent.system_prompt = (
            f"你是将领「{agent.name}」，当前执行战术「{result.selected_tactic.name}」。\n"
            f"战术描述：{result.selected_tactic.description}\n"
            f"行动序列：{' → '.join(result.selected_tactic.action_sequence[:3])}\n"
            f"作战风格：{result.selected_tactic.combat_style.value}\n"
            f"当前态势：{result.tactical_context.regime} "
            f"(战力比={result.tactical_context.strength_ratio:.2f})\n\n"
            f"{original or ''}"
        )
        try:
            output = agent.drum(task_description)
        finally:
            # ★ 必须还原：agents 在同一次 run 里跨任务复用，
            #   不还原会让上一条任务的战术泄漏进下一条的提示词。
            agent.system_prompt = original
        return output, agent.last_run_tokens

    def run_bingfu(self, task: BenchmarkTask, seed: int = 42,
                   ablation: str = "full", *,
                   tools_enabled: bool = True,
                   center_alignment: bool = False) -> TaskResult:
        """
        Run BingFu with tactic-driven agent selection.

        Args:
            task: Benchmark task
            seed: Random seed
            ablation: One of 'full', 'no_tactic', 'no_power', 'no_style', 'no_history'
        """
        t0 = time.time()
        tokens = 0
        tactic_name = ""
        tactic_alignment = 0.0

        # ★ 真的把 seed 用起来。
        #
        #   这个方法一直接受 `seed` 参数却**从不使用** —— no_tactic 和
        #   no_power 两个消融分支里的 random.choice 完全没被固定，
        #   同样的实验跑两次选到的将领不同、结果就不同。
        #   全仓库唯一一处 random.seed 在 statistical_test 的 bootstrap 里。
        #
        #   ★ 一个不可复现的消融组，没有资格出现在论文里 ——
        #     别人复现不出来，作者自己隔天也复现不出来。
        #
        #   用局部 Random 而不是 random.seed()：后者会改**全局**状态，
        #   顺手影响到调用方其它地方的随机性，那是很难查的耦合。
        #
        #   ★ 种子里混入 task.id：只用 seed 的话，每条任务都会选中同一位将领
        #     （同样的种子、同样的候选顺序）—— 那不是"随机分派"，
        #     是"固定分派给第一个"，消融的含义完全变了。
        rng = random.Random(f"{seed}::{task.id}")

        # ★★ 把 LLM 交给**任务评估器**。
        #
        #   此前四处 `TacticEngine()` 都是裸建的 —— LLM 挂给了将领，
        #   **从没挂给评估器**。于是 TaskAssessor 一直走无 LLM 的规则分支，
        #   对全部 20 条 benchmark 返回同一组值：
        #
        #       complexity_score = 5   （20/20 相同）
        #       enemy_power      = 72  （20/20 相同）
        #       required_capabilities  （20/20 相同）
        #
        #   而 5 维任务特征正是从这几个字段算出来的 ——
        #   实测：**五维里四维标准差为 0**，20 条任务只对应 3 个不同的向量，
        #   IR1（量子计算调研）和 C1（写 FastAPI 服务）拿到逐位相同的特征。
        #
        #   后果：所谓「任务自适应的战术选择」退化成常数函数 ——
        #   20 条任务里 19 条选出同一个战术（五火之变）、同一位将领（白起）。
        #   论文 Contribution 2 声称的 "rule-based + LLM-enhanced 双模式"，
        #   在实验里**只跑了规则那一半**。
        #
        #   ★ 挂上 LLM 之后评估器立刻有区分度（复杂度 7/6/6/7、
        #     敌方战力 120/108/108/132、所需能力各不相同）——
        #     所以这不是结构性空洞，是一处从来没接上的线。
        assessor = TaskAssessor(llm_provider=self.llm) if self.llm else None

        # ★ 按任务类型装配 VERIFY 门禁。
        #
        #   实测：Code 12%、Write 38%，而 IR 是 100%。
        #   差别不在模型，在于 benchmark 里 tools=None 让 ReAct 循环
        #   恒定只跑一轮 —— 单次生成一个全栈应用，不执行不检查不重试。
        #
        #   门禁做的是**可证伪**的检查（语法能否 ast.parse、
        #   数字能否溯源、结构要素是否齐全），不是"再问模型一次好不好"。
        #   后者不会失败，因而等于没有。
        # ★ 按任务类型取执行计划 —— 每一项都由实测数据决定。
        #
        #   分类别成功率（n≈39/组）：
        #       Code 12% / Write 38% / Data 75% / Reason 86% / IR 100%
        #   战术注入的效应：
        #       Data +50、IR +50；**Code −26、Write −12**
        #   门禁的 token 代价：约 4–8 倍
        #
        #   所以：Code/Write 关战术开门禁，IR/Reason 开战术关门禁。
        #   这不是加功能，是让已有能力只用在它有效的地方 ——
        #   一套统一流水线对这五类任务本来就是错的。
        plan = route_for(task.category)

        # ★ 装配工具带 —— 这是 Code 类 12–25% 的直接解药。
        #
        #   在此之前将领拿到的是 tools=None，于是 has_tool_calls 恒为假，
        #   ReAct 循环**只跑一轮就返回**：模型被要求一次性写出
        #   带鉴权、异步数据库的服务，不执行、不看报错、不修改。
        #
        #   给它一个能真的跑起来的解释器，才谈得上"迭代"。
        # ★ 工具开关是为了做**有工具 vs 无工具**的对照。
        #
        #   默认开启 —— 无工具是被测的旧行为，不该是默认。
        belt = belt_for(task.category)
        if not tools_enabled:
            belt.enable_web = False
            belt.enable_code = False

        def _verify(output: str):
            # ★ 来源集来自**工具实际抓回的原文**，不再是任务描述本身。
            #
            #   check_numbers_traceable 之前被迫禁用，理由是
            #   "将领没有工具，研究类任务的数字无处可溯"。
            #   现在工具抓回的资料进了 SourceStore，那条断言重新成立。
            #
            #   注意 sources 只在工具真被调用后才非空 —— 若模型
            #   一次工具都没调，来源集为空，溯源检查会自动跳过，
            #   而不是把"没有证据"误判成"数字全是编的"。
            return verify_output(
                output,
                category=task.category,
                criteria=task.evaluation_criteria,
                sources=belt.sources() or [task.description],
            )

        try:
            # Create agents from presets
            agents = {}
            for name in ["韩信", "白起", "诸葛亮", "项羽"]:
                profile = get_preset(name)
                agent = Agent(
                    name=name, role="将军", profile=profile,
                    verify_fn=(_verify if (self.llm and plan.verify_enabled) else None),
                    max_revisions=plan.max_revisions,
                    max_output_tokens=plan.max_output_tokens,
                )
                # 把工具真正挂到将领身上
                for _tname, _tfunc in belt.as_functions().items():
                    agent.add_tool(_tname, _tfunc)
                if self.llm:
                    agent.llm = self.llm
                agents[name] = agent

            if ablation == "no_tactic":
                # ★ 只去掉「战术」这一件事：将领仍按战力选，只是不注入战术。
                #
                #   原实现是「随机选将领 + 不注入战术」—— 一次改了两个变量。
                #   于是 full 领先的部分无法归因到任何一个组件。
                agent = max(
                    agents.values(),
                    key=lambda a: a.profile.stats.total_power() if a.profile else 250,
                )
                tactic_name = "none"
                if self.llm:
                    output = agent.drum(task.description)
                    tokens = agent.last_run_tokens
                else:
                    output = f"[BingFu no_tactic for {task.id}]"
                    tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来
                # ★ 这里原本被一句 `tactic_name = "random"` 覆盖掉。
                #
                #   这一档**根本没有使用任何战术**，报表却写成用了一个
                #   叫 "random" 的战术。后果不是名字难看，而是任何
                #   按 tactic_used 分组的分析都会凭空多出一个不存在的战术，
                #   并把"未使用战术"的样本算进"使用了某战术"里。
                #
                #   保持 "none" —— 没有就是没有。
                tactic_name = "none"

            elif ablation == "no_power":
                # ★ 只去掉「按战力选将领」这一件事：战术照选、**照注入**，将领随机。
                #
                #   原实现算出战术后**丢掉不注入**（只存进 tactic_name 用于报表），
                #   于是它与 no_tactic 在操作上完全相同 ——
                #   实测 29% vs 28%，p = 1.0，统计上无法区分。
                #   那不是一档消融，是同一档跑了两遍。
                tactic_engine = TacticEngine(assessor=assessor,
                                             center_alignment=center_alignment)
                result = tactic_engine.select_tactic(task.description, agents)
                agent = rng.choice(list(agents.values()))
                tactic_name = result.selected_tactic.name
                tactic_alignment = result.alignment_score
                if self.llm:
                    output, tokens = self._run_with_tactic(agent, task.description, result)
                else:
                    output = f"[BingFu no_power for {task.id}]"
                    tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来

            elif ablation == "random_tactic":
                # ★ 缺失的那个对照：将领按战力选（与 full 相同），
                #   但注入一个**随机抽取**的战术。
                #
                #   为什么必须有这一档：
                #
                #   full 与 no_tactic 的差距（阶段性观察 50% vs 35.7%）
                #   同时包含两种可能的来源：
                #     ① 选出的战术**确实适配**这个任务
                #     ② 提示词里多了约 580 token 的结构化指导
                #        （行动序列、作战风格、态势描述）
                #
                #   ★ 只要没有这一档，这两者就分不开 ——
                #     而它们对论文的意义完全不同：
                #     ① 支持"战术选择有效"这个核心主张；
                #     ② 只说明"更详细的提示词更有用"，
                #        那是提示工程的常识，不是这篇论文的贡献。
                #
                #   随机战术 vs 选中战术，注入格式完全一致、长度量级相同，
                #   两者之差才是**战术选择本身**的贡献。
                tactic_engine = TacticEngine(assessor=assessor)
                result = tactic_engine.select_tactic(task.description, agents)
                agent = max(
                    agents.values(),
                    key=lambda a: a.profile.stats.total_power() if a.profile else 250,
                )
                # 用同一个 rng（已按 seed::task_id 固定）随机换掉选中的战术
                library = list(get_tactic_library().values())
                result.selected_tactic = rng.choice(library)
                tactic_name = f"random:{result.selected_tactic.name}"
                tactic_alignment = 0.0
                if self.llm:
                    output, tokens = self._run_with_tactic(agent, task.description, result)
                else:
                    output = f"[BingFu random_tactic for {task.id}]"
                    tokens = 0

            elif ablation == "no_style":
                # Power matching but no style bonus
                tactic_engine = TacticEngine(weights=(0.0, 0.8, 0.2), assessor=assessor)
                result = tactic_engine.select_tactic(task.description, agents)
                agent = agents.get(result.selected_agent_name, list(agents.values())[0])
                tactic_name = result.selected_tactic.name
                tactic_alignment = result.alignment_score
                if self.llm:
                    output = agent.drum(task.description)
                    tokens = agent.last_run_tokens   # ★ 真实用量，不再写死 1200
                else:
                    output = f"[BingFu no_style for {task.id}]"
                    tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来

            elif ablation == "no_history":
                # Full matching but no history component
                tactic_engine = TacticEngine(weights=(0.4, 0.6, 0.0), assessor=assessor)
                result = tactic_engine.select_tactic(task.description, agents)
                agent = agents.get(result.selected_agent_name, list(agents.values())[0])
                tactic_name = result.selected_tactic.name
                tactic_alignment = result.alignment_score
                if self.llm:
                    output = agent.drum(task.description)
                    tokens = agent.last_run_tokens   # ★ 真实用量，不再写死 1200
                else:
                    output = f"[BingFu no_history for {task.id}]"
                    tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来

            else:  # full BingFu
                tactic_engine = TacticEngine(assessor=assessor)
                result = tactic_engine.select_tactic(task.description, agents)

                # Check if LLM is available; if so, execute properly
                agent = agents.get(result.selected_agent_name)
                if agent is None:
                    agent = max(agents.values(),
                               key=lambda a: a.profile.stats.total_power() if a.profile else 250)

                tactic_name = result.selected_tactic.name
                tactic_alignment = result.alignment_score

                if self.llm:
                    # ★ 按分流决定注不注入战术。
                    #
                    #   实测：战术注入在 Data +50、IR +50，
                    #   但在 **Code −26、Write −12** —— 兵法提示把模型推向
                    #   "分析、侦察、评估态势"的框架，对信息类任务对路，
                    #   对写代码和写作是干扰。
                    #
                    #   这里**不是把功能砍掉**，是让它只用在有效的地方。
                    #   对所有任务一视同仁地注入，等于用两类任务的损失
                    #   去换另两类的收益，而总账是不是正的从来没人算过。
                    #
                    #   注入路径与 no_power 共用同一个函数，措辞逐字相同 ——
                    #   否则措辞差异会混进消融效应，而那种偏差无法在结果上分辨。
                    if plan.inject_tactic:
                        output, tokens = self._run_with_tactic(
                            agent, task.description, result
                        )
                    else:
                        tactic_name = f"{result.selected_tactic.name}(未注入)"
                        output = agent.drum(task.description)
                        tokens = agent.last_run_tokens
                else:
                    output = f"[BingFu full for {task.id}] " \
                             f"Tactic: {tactic_name}, Agent: {agent.name}"
                    tokens = 0   # ★ 无 LLM 的占位路径：0 = 没测到，不是编一个 800 出来

            # Evaluate with LLM judge
            judge = LLMJudge(self.llm)
            success, score, eval_details = judge.evaluate(task, output)

            # Calculate agent utilization
            util = 0.85 if ablation == "full" else 0.7

            # ★ 把验收轨迹带出来。没有它，「门禁让 Code 类变好了」
            #   这句话无法证伪 —— 成功率上升也可能只是模型当天状态好。
            trace = list(getattr(agent, "_last_verify_trace", []) or [])
            last = trace[-1] if trace else {}

            return TaskResult(
                task_id=task.id, method=f"BingFu ({ablation})", seed=seed,
                success=success, completion_score=score, tokens_consumed=tokens,
                agent_utilization=util, tactic_used=tactic_name,
                tactic_alignment=tactic_alignment,
                execution_time=time.time()-t0, output=output[:500],
                verify_revisions=max(0, len(trace) - 1),
                verify_passed=bool(last.get("passed", True)),
                verify_stopped_by=str(last.get("stopped_by", "")),
                verify_checks=list(last.get("checks", [])),
                tool_calls=dict(belt.call_counts),
                output_len=len(output or ""),
            )

        except Exception as e:
            return TaskResult(
                task_id=task.id, method=f"BingFu ({ablation})", seed=seed,
                success=False, completion_score=1.0, tokens_consumed=500,
                agent_utilization=0.0, execution_time=time.time()-t0, error=str(e),
            )


# ═══════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════

BINGFU_MAX_REVISIONS = 2
"""VERIFY 不通过时最多回炉几次。

★ 2 是在"能捞回多少"与"成本翻几倍"之间取的：
  语法错这类确定性失败通常一轮就能修好；
  两轮还修不好的，多半是任务本身超出了单次生成的能力，
  再回炉只是烧钱。
"""


CHECKPOINT_EVERY = 25
"""每多少次运行落一次检查点。

★ 25 次约合 10 分钟 —— 最坏情况下丢掉的调用量控制在十分钟以内。
  这个数是在"写盘开销"和"崩溃损失"之间取的：写得太频繁会拖慢实验，
  太稀疏则失去意义。
"""


MAX_INFRA_FAILURE_RATE = 0.05
"""基础设施故障率的上限 —— 超过它就拒绝落盘。

★ 5% 是个判断，不是定理：低于它，剔除少数缺失点对结论的影响
  小于重跑一整轮的代价；高于它，说明环境本身有问题，
  这批数据的可信度已经不是「剔除几条」能挽救的。
"""


class LLMCallFailed(RuntimeError):
    """一次 LLM 调用在**基础设施层面**失败了（连不上、认证错、超时、限流）。

    ════════════════════════════════════════════════════════════
     ★ 为什么必须把它和「任务失败」分开
    ════════════════════════════════════════════════════════════

    provider 在调用失败时**不抛异常**，而是返回一个正常的 LLMResponse，
    content 是 "❌ OpenAI 调用失败：Connection error."，
    并把 finish_reason 置为 "error"。

    那个 finish_reason 一直设着，而**全仓库没有任何地方读它**。

    于是在实验框架里发生的事情是：

        调用失败 → output = "❌ OpenAI 调用失败：..."（35 字符）
                 → 没有异常，error 字段为空
                 → 裁判按输出长度打分（<50 判失败）
                 → **记录成一次「这个方法没能完成任务」**

    也就是说：**一次网络抖动和一个答不出题的模型，在数据里完全一样**。
    实测首次运行时 7 个方法全线 0.0% —— 那不是模型不行，是调用就没成。

    ★ 一批混入了连接失败的成功率，是**不可解释**的数字。
      拿它去支撑「BingFu 比 MetaGPT 高 10.4 个点」这类结论，
      结论的正确与否完全取决于当天的网络。
    """


def _parse_judge_json(content: str) -> Dict[str, Any]:
    """从模型回复里取出**最外层**的 JSON 对象。

    ★ 关键在「最外层」。裁判的返回是嵌套结构：

        {"success": ..., "completion_score": ..., "criteria_scores": {...}, ...}

      取到内层的 criteria_scores，读出来的 success/completion_score
      永远是默认值 —— 而那看起来和一次正常的评估**完全一样**。

    ★ 解析失败就抛，让调用方决定降级 —— 不在这里静默返回空字典。
      返回 `{}` 会让上层拿到默认的 (False, 3.0)，
      也就是把"没解析出来"伪装成"评估结果是失败"。
    """

    text = content.strip()

    # 模型常把 JSON 包在 ```json ... ``` 里
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 回复里混了散文时：从第一个 { 起做**括号配平**扫描，取最外层对象。
    start = text.find("{")
    if start == -1:
        raise ValueError(f"裁判回复里没有 JSON 对象：{text[:200]}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("裁判回复的最外层不是 JSON 对象")
                return parsed

    raise ValueError(f"裁判回复里的 JSON 括号不配平：{text[:200]}")


def _generate_checked(llm, *args, **kwargs):
    """调 LLM，**基础设施故障就地终止**，不伪装成正常回复。

    ★ 这就是那个从来没有消费方的 finish_reason 的消费方。
    """

    response = llm.generate(*args, **kwargs)
    reason = getattr(response, "finish_reason", None)

    if reason == "error":
        raise LLMCallFailed(response.content or "LLM 调用失败（无详情）")

    # ★ 'length' 也必须拦，而且要单独报。
    #
    #   裁判的回复是 JSON，max_tokens=1024。一旦被截断，
    #   JSON 就是残缺的 → 解析失败 → 落进 except Exception
    #   → **静默降级成启发式打分**（按输出长度给分）。
    #
    #   于是报告上照样有一整套成功率数字，
    #   而实际用的评分方法与论文所述的 LLM-as-judge 不是一回事，
    #   且没有任何地方说明发生过降级。
    #
    # ★ 这与 Agent 那边的截断是同一个 bug 的两个位置：
    #   API 明确说了"我被截断了"，而代码只看它等不等于 'error'。
    #   修一处不够 —— 同一个盲点会在每一个读 finish_reason
    #   的地方重复出现。
    if reason == "length":
        raise LLMCallFailed(
            "LLM 回复被长度上限截断（finish_reason='length'），"
            "结构化输出不完整。这属于**没测成**，不是模型答得不好。"
            f"已生成 {len(response.content or '')} 字符。"
        )
    return response


class ExperimentRunner:
    """
    Complete experiment pipeline.

    Usage:
        runner = ExperimentRunner(llm_provider=my_llm)
        results = runner.run_all(seeds=[42, 123, 456])
        runner.save_results(results, "experiment_results.json")
        runner.print_report(results)
    """

    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self.baseline = BaselineRunner(llm_provider)
        self.tasks = BENCHMARK_TASKS

    def run_all(
        self,
        methods: Optional[List[str]] = None,
        seeds: List[int] = None,
        tasks: Optional[List[BenchmarkTask]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, AggregateMetrics]:
        """
        Run full experiment suite.

        Args:
            methods: Methods to run. Default: all.
            seeds: Random seeds. Default: [42].
            tasks: Tasks to run. Default: all BENCHMARK_TASKS.

        Returns:
            {method_name: AggregateMetrics}
        """
        if seeds is None:
            seeds = [42]
        if tasks is None:
            tasks = self.tasks
        if methods is None:
            methods = ["GPT-4 Solo", "AutoGen", "CrewAI", "MetaGPT",
                       "BingFu (full)", "BingFu (no_tactic)", "BingFu (no_power)"]

        all_results: Dict[str, List[TaskResult]] = {m: [] for m in methods}

        total = len(tasks) * len(seeds) * len(methods)
        completed = 0
        t_start = time.time()

        for seed in seeds:
            for task in tasks:
                for method in methods:
                    completed += 1
                    if completed % 10 == 0:
                        # ★ 打上时间戳与单次耗时。
                        #
                        #   原先只打 "Progress: N/M"。一轮 700 次的实验在
                        #   第 410 次挂死，进度停了 8 小时 20 分钟 ——
                        #   而从输出上**完全看不出它是卡住了还是变慢了**，
                        #   因为没有任何时间信息可比。
                        #
                        #   一行时间戳就能让"停了八小时"一眼可见。
                        elapsed = time.time() - t_start
                        rate = elapsed / completed
                        remain = rate * (total - completed)
                        print(
                            f"  Progress: {completed}/{total} runs"
                            f"  [{time.strftime('%H:%M:%S')}]"
                            f"  均 {rate:.1f}s/次  预计剩余 {remain / 60:.0f} 分钟",
                            flush=True,
                        )

                    if method == "GPT-4 Solo":
                        result = self.baseline.run_gpt4_solo(task, seed)
                    elif method == "AutoGen":
                        result = self.baseline.run_autogen_style(task, seed)
                    elif method == "CrewAI":
                        result = self.baseline.run_crewai_style(task, seed)
                    elif method == "MetaGPT":
                        result = self.baseline.run_metagpt_style(task, seed)
                    elif method.startswith("BingFu"):
                        ablation = method.replace("BingFu (", "").replace(")", "")
                        result = self.baseline.run_bingfu(task, seed, ablation)
                    else:
                        continue

                    all_results[method].append(result)

                    # ★★ 增量检查点 —— 每 CHECKPOINT_EVERY 次把原始记录落盘。
                    #
                    #   原实现只在**全部跑完**之后写一次结果文件。
                    #   这个项目已经因此丢了两批数据：
                    #     · 一轮 300 次全部跑完，落盘时被过严的防线拒绝 → 明细全失
                    #     · 一轮 700 次跑到 410 次挂死 8 小时 → 全失
                    #
                    #   ★ 长任务把全部产出押在最后一次写盘上，
                    #     等于让「跑完」和「拿到数据」变成同一个不可分的赌注。
                    #     中途任何一次异常，前面所有调用的钱都白花。
                    #
                    #   检查点是原始 TaskResult，不是聚合值 ——
                    #   聚合可以随时重算，原始记录丢了就再也回不来。
                    if checkpoint_path and completed % CHECKPOINT_EVERY == 0:
                        try:
                            self._write_checkpoint(checkpoint_path, all_results, completed, total)
                        except Exception as exc:      # 检查点失败不能拖垮实验本身
                            print(f"  !! 检查点写入失败（实验继续）：{exc}", flush=True)

        # 收尾再写一次，保证最后不足一个间隔的记录也落盘
        if checkpoint_path:
            try:
                self._write_checkpoint(checkpoint_path, all_results, completed, total)
            except Exception as exc:
                print(f"  !! 收尾检查点写入失败：{exc}", flush=True)

        # Aggregate
        metrics = {}
        for method, results in all_results.items():
            metrics[method] = self._aggregate(method, results)

        return metrics

    def run_quick_pilot(
        self,
        n_tasks: int = 5,
        seeds: List[int] = None,
    ) -> Dict[str, AggregateMetrics]:
        """Quick pilot: run first n_tasks with 1 seed."""
        if seeds is None:
            seeds = [42]
        pilot_tasks = self.tasks[:n_tasks]
        return self.run_all(
            methods=["GPT-4 Solo", "AutoGen", "MetaGPT", "BingFu (full)", "BingFu (no_tactic)"],
            seeds=seeds,
            tasks=pilot_tasks,
        )

    @staticmethod
    def _write_checkpoint(path, all_results, completed: int, total: int) -> None:
        """把当前已有的原始记录写到检查点文件（原子替换）。

        ★ 先写临时文件再 os.replace —— 直接覆写的话，
          写到一半崩溃会把上一份完好的检查点也毁掉，
          那就比不写检查点更糟。
        """

        payload = {
            "_checkpoint": {
                "completed": completed,
                "total": total,
                "finished": completed >= total,
            },
            "results": {
                method: [r.to_dict() for r in rows]
                for method, rows in all_results.items()
            },
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)

    def _aggregate(
        self, method: str, results: List[TaskResult]
    ) -> AggregateMetrics:
        """Aggregate results into metrics."""
        if not results:
            return AggregateMetrics(method=method, avg_sr=0, std_sr=0,
                                    avg_tcs=0, std_tcs=0, avg_tokens=0,
                                    avg_utilization=0)

        srs = [1.0 if r.success else 0.0 for r in results]
        tc_scores = [r.completion_score for r in results]
        tokens = [r.tokens_consumed for r in results]
        utils = [r.agent_utilization for r in results]

        # Per-category SR
        per_cat = {}
        for cat in ["IR", "Code", "Data", "Write", "Reason"]:
            cat_results = [r for r in results
                          if any(t.id.startswith(cat[0]) for t in self.tasks
                                if t.id == r.task_id)]
            if cat_results:
                per_cat[cat] = sum(1.0 if r.success else 0.0 for r in cat_results) / len(cat_results)

        return AggregateMetrics(
            method=method,
            avg_sr=np.mean(srs) if srs else 0,
            std_sr=np.std(srs) if srs else 0,
            avg_tcs=np.mean(tc_scores) if tc_scores else 0,
            std_tcs=np.std(tc_scores) if tc_scores else 0,
            avg_tokens=np.mean(tokens) if tokens else 0,
            avg_utilization=np.mean(utils) if utils else 0,
            per_category_sr=per_cat,
            per_task_results=results,
        )

    def statistical_test(
        self, results_a: List[TaskResult], results_b: List[TaskResult],
        n_bootstrap: int = 10000,
    ) -> Dict[str, float]:
        """
        Bootstrap test for significance between two methods.
        Returns p-value and confidence intervals.
        """
        srs_a = np.array([1.0 if r.success else 0.0 for r in results_a])
        srs_b = np.array([1.0 if r.success else 0.0 for r in results_b])

        observed_diff = np.mean(srs_a) - np.mean(srs_b)

        # Bootstrap
        diffs = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(srs_a, size=len(srs_a), replace=True)
            sample_b = np.random.choice(srs_b, size=len(srs_b), replace=True)
            diffs.append(np.mean(sample_a) - np.mean(sample_b))

        diffs = np.array(diffs)
        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)

        return {
            "observed_diff": observed_diff,
            "p_value": p_value,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "significant_at_05": p_value < 0.05,
        }

    def print_report(self, metrics: Dict[str, AggregateMetrics]):
        """Print formatted experiment report."""
        print("\n" + "=" * 80)
        print("BINGFU EXPERIMENT REPORT")
        print("=" * 80)

        # ★ 基础设施故障必须**排在数字前面**说。
        #
        #   放在报告末尾或者不说，读表的人会把 0.0% 当成「这个方法不行」，
        #   而真相可能是一次调用都没成。实测首次运行时全线 0.0%，
        #   真因就是调用失败被当成了任务失败。
        failures = self.count_infra_failures(metrics)
        total_failures = sum(failures.values())
        if total_failures:
            print("")
            print("  !! 本次有 %d 次运行是**基础设施故障**（连不上/认证/超时/限流）" % total_failures)
            for method, count in sorted(failures.items()):
                if count:
                    print(f"       {method}: {count} 次")
            print("  !! 下面的成功率**掺了这些失败**，不可用于任何结论。")
            print("")

        print(f"\n{'Method':<22} {'Avg SR':>8} {'±Std':>6} {'TCS':>6} {'Tokens':>8} {'Util':>6}")
        print("-" * 60)

        # Sort by SR descending
        sorted_methods = sorted(metrics.values(), key=lambda m: m.avg_sr, reverse=True)
        for m in sorted_methods:
            print(f"{m.method:<22} {m.avg_sr:>7.1%} {m.std_sr:>5.1%} "
                  f"{m.avg_tcs:>5.2f} {m.avg_tokens:>7.0f}K {m.avg_utilization:>5.1%}")

        print(f"\n{'─'*80}")
        print("Per-Category Success Rate:")
        print(f"{'Method':<22} {'IR':>8} {'Code':>8} {'Data':>8} {'Write':>8} {'Reason':>8}")
        print("-" * 62)
        for m in sorted_methods:
            cats = m.per_category_sr
            print(f"{m.method:<22} "
                  f"{cats.get('IR', 0):>7.1%} {cats.get('Code', 0):>7.1%} "
                  f"{cats.get('Data', 0):>7.1%} {cats.get('Write', 0):>7.1%} "
                  f"{cats.get('Reason', 0):>7.1%}")

        # Statistical tests
        bingfu_full = metrics.get("BingFu (full)")
        metagpt = metrics.get("MetaGPT")
        if bingfu_full and metagpt:
            test = self.statistical_test(
                bingfu_full.per_task_results, metagpt.per_task_results
            )
            print(f"\n{'─'*80}")
            print(f"BingFu vs MetaGPT: ΔSR={test['observed_diff']:.1%}, "
                  f"p={test['p_value']:.4f}, significant={test['significant_at_05']}")

        print("=" * 80)

    @staticmethod
    def count_infra_failures(metrics: Dict[str, AggregateMetrics]) -> Dict[str, int]:
        """统计每个方法有多少次运行是**基础设施故障**（非任务失败）。"""

        counts: Dict[str, int] = {}
        for method, m in metrics.items():
            counts[method] = sum(1 for r in m.per_task_results if r.error)
        return counts

    def save_results(self, metrics: Dict[str, AggregateMetrics], path: str):
        """Save experiment results to JSON.

        ★ 有基础设施故障时**拒绝落盘**。

          `figures/generate_figures.py` 判断「有没有实测数据」的唯一依据
          就是这个文件存不存在 —— 文件一出现，占位水印就消失。

          所以一份混入了连接失败的结果文件，会让图**看起来是实测的**，
          而那比原来的占位数据更危险：占位数据至少还有人记得它是占位的。

          「跑不了」要报「跑不了」，不能报「通过」。
        """

        failures = self.count_infra_failures(metrics)
        total_failures = sum(failures.values())
        total_runs = sum(len(m.per_task_results) for m in metrics.values()) or 1
        failure_rate = total_failures / total_runs

        # ★ 阈值化，而不是「有一次就全废」。
        #
        #   第一版写的是「只要有基础设施故障就拒绝落盘」。实际后果：
        #   一轮 300 次的实验全部跑完，因为其中 **2 次**瞬时 API 故障，
        #   整份数据被拒绝写入、进程以退出码 1 结束 —— 三百次调用的
        #   per-task 明细全部丢失，只剩终端上的汇总数字。
        #
        #   ★ 防线的方向是对的（掺了连接失败的数据不可解释），
        #     但把「2/300」与「300/300」按同一条规则处理是过度反应。
        #     瞬时故障是**缺失数据点**，标准处理是剔除并记录数量，
        #     而不是作废整批 —— 那等于用一条防线制造了更大的损失。
        #
        #   现在：低于阈值 → 已在 _aggregate 里剔除，落盘并记下剔了几条；
        #        高于阈值 → 说明环境本身有问题，拒绝。
        if failure_rate > MAX_INFRA_FAILURE_RATE:
            detail = "、".join(f"{m}:{n}" for m, n in failures.items() if n)
            raise RuntimeError(
                f"拒绝写入结果文件：{total_failures}/{total_runs} 次运行是**基础设施故障**"
                f"（{failure_rate:.1%}，超过阈值 {MAX_INFRA_FAILURE_RATE:.0%}）（{detail}）。\n"
                f"  这些不是「方法没完成任务」，是调用压根没成 —— "
                f"这个比例下混进成功率会让整份数据不可解释。\n"
                f"  先把 LLM 调用修通再重跑。"
            )

        # ★ 剔除了多少必须写进文件本身。
        #   只在终端上说一句，文件被别人拿走之后这条信息就没了 ——
        #   而"剔除了 2 条"和"一条都没剔"在数据上是两回事。
        data: Dict[str, Any] = {
            "_meta": {
                "total_runs": total_runs,
                "infra_failures_excluded": total_failures,
                "infra_failure_rate": round(failure_rate, 4),
                "per_method_excluded": {m: n for m, n in failures.items() if n},
            }
        }
        for method, m in metrics.items():
            data[method] = {
                "aggregate": m.to_dict(),
                "per_task": [r.to_dict() for r in m.per_task_results],
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {path}")

    def load_results(self, path: str) -> Dict[str, AggregateMetrics]:
        """Load experiment results from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics = {}
        for method, mdata in data.items():
            results = []
            for rdata in mdata.get("per_task", []):
                results.append(TaskResult(**rdata))
            agg_data = mdata.get("aggregate", {})
            metrics[method] = AggregateMetrics(
                method=method,
                avg_sr=agg_data.get("avg_sr", 0),
                std_sr=agg_data.get("std_sr", 0),
                avg_tcs=agg_data.get("avg_tcs", 0),
                std_tcs=agg_data.get("std_tcs", 0),
                avg_tokens=agg_data.get("avg_tokens", 0),
                avg_utilization=agg_data.get("avg_utilization", 0),
                per_category_sr=agg_data.get("per_category_sr", {}),
                per_task_results=results,
            )
        return metrics


# ═══════════════════════════════════════════════════════════════
# Quick Demo
# ═══════════════════════════════════════════════════════════════

def demo_experiment():
    """Demonstrate experiment framework without LLM (heuristic eval)."""
    print("=" * 60)
    print("BingFu Experiment Framework — Quick Demo")
    print("=" * 60)
    print(f"Tasks loaded: {len(BENCHMARK_TASKS)}")

    runner = ExperimentRunner(llm_provider=None)

    # Run pilot with heuristic evaluation
    print("\nRunning pilot (5 tasks, 1 seed, heuristic eval)...")
    metrics = runner.run_quick_pilot(n_tasks=5, seeds=[42])

    runner.print_report(metrics)

    # Save results
    runner.save_results(metrics, "pilot_results.json")

    return metrics


if __name__ == "__main__":
    demo_experiment()
