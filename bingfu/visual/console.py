"""
BingFu 中军帐主控制台
古代军事风格的Multi-Agent可视化监控界面

v0.4.0: 新增 LLM 驱动的自然语言理解
当配置了 LLM Provider 后，自然语言指令将走 LLM 解析而非关键词匹配。
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .styles import COLORS, FONTS, ICONS
from .components import (
    GeneralCard, BattleStatusPanel, ReportPanel,
    StatsBar, CommandInput, StyledFrame
)
from ..commander import Commander
from ..memory import Memory


class MilitaryCommandConsole:
    """
    中军帐可视化控制台

    提供古代军事风格的Multi-Agent框架可视化界面，
    支持将领状态监控、战役态势分析、军情速递等功能。

    使用示例:
    ```python
    from bingfu.visual import MilitaryCommandConsole

    # 创建控制台
    console = MilitaryCommandConsole()

    # 添加将领
    console.add_general("韩信", "online", "统帅", "正在分析战场形势")

    # 添加军情报告
    console.add_report("侦察回报", "发现敌军粮草运输队", "success")

    # 启动
    console.run()
    ```
    """

    def __init__(
        self,
        title: str = "兵符 · 中军帐",
        width: int = 1200,
        height: int = 800,
        commander: Optional[Commander] = None,
        memory: Optional[Memory] = None,
        llm_provider: Optional[Any] = None,
        bingfu_instance: Optional[Any] = None,
    ):
        """
        初始化控制台

        Args:
            title: 窗口标题
            width: 窗口宽度
            height: 窗口高度
            commander: 可选的Commander实例，用于与框架集成
            memory: 可选的Memory实例，用于状态持久化
            llm_provider: 可选的LLM Provider实例，用于自然语言理解
            bingfu_instance: 可选的BingFu实例，用于框架集成
        """
        self.title = title
        self.width = width
        self.height = height
        self.commander = commander
        self.memory = memory
        self.llm_provider = llm_provider
        self.bingfu_instance = bingfu_instance

        # 内部状态
        self.generals: Dict[str, Dict] = {}
        self.reports: List[Dict] = []
        self.is_running = False

        # 命令历史
        self.command_history: List[str] = []
        self.history_index = -1

        # 创建窗口
        self._root: Optional[tk.Tk] = None
        self._create_window()

    def _create_window(self):
        """创建主窗口"""
        self._root = tk.Tk()
        self._root.title(self.title)
        self._root.geometry(f"{self.width}x{self.height}")
        self._root.minsize(900, 600)

        # 设置主题色
        self._root.configure(bg=COLORS["bg_dark"])

        # 阻止窗口关闭时自动销毁
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 构建UI
        self._build_ui()

    def _build_ui(self):
        """构建用户界面"""
        # 标题栏
        self._create_title_bar()

        # ══════════════════════════════════════════════
        # 底部固定元素 — 先 pack 确保始终可见
        # ══════════════════════════════════════════════

        # 底部状态栏（最底部）
        self.stats_bar = StatsBar(self._root)
        self.stats_bar.pack(side="bottom", fill="x")

        # 命令输入区（军令输入框）
        self.command_input = CommandInput(
            self._root,
            on_submit=self._handle_command
        )
        self.command_input.pack(side="bottom", fill="x", pady=(0, 5))

        # ══════════════════════════════════════════════
        # 可拖拽分割区 — 主内容(上) + 日志区(下)
        # ══════════════════════════════════════════════

        v_paned = tk.PanedWindow(
            self._root,
            orient="vertical",
            bg=COLORS["bg_dark"],
            sashrelief="groove",
            sashwidth=4,
            handlesize=8
        )
        v_paned.pack(fill="both", expand=True)

        # 上半部分：三栏主内容区
        main_paned = tk.PanedWindow(
            v_paned,
            bg=COLORS["bg_dark"],
            sashrelief="groove",
            sashwidth=5,
            handlesize=10,
            handlepad=5
        )
        v_paned.add(main_paned, height=420)

        # 左侧面板 - 将领名录
        left_panel = self._create_left_panel(main_paned)
        main_paned.add(left_panel, width=280)

        # 中间面板 - 战役态势
        center_panel = self._create_center_panel(main_paned)
        main_paned.add(center_panel, width=400)

        # 右侧面板 - 军情速递
        right_panel = self._create_right_panel(main_paned)
        main_paned.add(right_panel)

        # 下半部分：日志输出区（可拖拽调整高度）
        log_container = tk.Frame(v_paned, bg=COLORS["bg_dark"])
        v_paned.add(log_container, height=220)

        # 日志标题栏
        log_header = tk.Frame(log_container, bg=COLORS["bg_medium"], height=28)
        log_header.pack(fill="x", padx=5, pady=(2, 0))
        log_header.pack_propagate(False)

        tk.Label(
            log_header,
            text=f"  {ICONS['report']} 战报传回",
            font=FONTS["small"],
            bg=COLORS["bg_medium"],
            fg=COLORS["gold"],
            anchor="w"
        ).pack(side="left", padx=5, pady=3)

        # 日志滚动文本框
        self.log_area = scrolledtext.ScrolledText(
            log_container,
            font=FONTS["mono"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["gold"],
            relief="flat",
            state="disabled",
            wrap="word"
        )
        self.log_area.pack(fill="both", expand=True, padx=5, pady=(2, 5))

        # 初始化日志
        self._log("系统启动中...")

    def _create_title_bar(self):
        """创建标题栏"""
        title_frame = tk.Frame(self._root, bg=COLORS["bg_dark"], height=50)
        title_frame.pack(fill="x", pady=(0, 5))
        title_frame.pack_propagate(False)

        # 左侧装饰
        left_decor = tk.Label(
            title_frame,
            text="╔══╗",
            font=("Consolas", 12),
            bg=COLORS["bg_dark"],
            fg=COLORS["gold"]
        )
        left_decor.pack(side="left", padx=(20, 5))

        # 标题
        title_label = tk.Label(
            title_frame,
            text=self.title,
            font=FONTS["title"],
            bg=COLORS["bg_dark"],
            fg=COLORS["gold"]
        )
        title_label.pack(side="left")

        # 右侧装饰
        right_decor = tk.Label(
            title_frame,
            text="╔══╗",
            font=("Consolas", 12),
            bg=COLORS["bg_dark"],
            fg=COLORS["gold"]
        )
        right_decor.pack(side="right", padx=(5, 20))

        # 副标题
        subtitle_label = tk.Label(
            title_frame,
            text="Multi-Agent 战役指挥系统",
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_muted"]
        )
        subtitle_label.pack(side="right", padx=(10, 5))

    def _create_left_panel(self, parent) -> tk.Frame:
        """创建左侧面板 - 将领名录"""
        frame = tk.Frame(parent, bg=COLORS["bg_dark"])

        # 标题
        header = tk.Frame(frame, bg=COLORS["gold_dark"], height=35)
        header.pack(fill="x", padx=5, pady=(5, 0))
        header.pack_propagate(False)

        tk.Label(
            header,
            text=f"  {ICONS['general']} 将领名录  ",
            font=FONTS["subtitle"],
            bg=COLORS["gold_dark"],
            fg=COLORS["bg_dark"]
        ).pack(side="left", pady=5)

        # 滚动区域
        canvas_frame = tk.Frame(frame, bg=COLORS["bg_dark"])
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(canvas_frame)
        scrollbar.pack(side="right", fill="y")

        self.generals_canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS["bg_dark"],
            highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.generals_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.generals_canvas.yview)

        self.generals_frame = tk.Frame(self.generals_canvas, bg=COLORS["bg_dark"])
        self.generals_canvas.create_window((0, 0), window=self.generals_frame, anchor="nw")

        self.generals_frame.bind("<Configure>",
            lambda e: self.generals_canvas.configure(scrollregion=self.generals_canvas.bbox("all")))

        # 操作按钮
        btn_frame = tk.Frame(frame, bg=COLORS["bg_dark"])
        btn_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(
            btn_frame,
            text=f"{ICONS['drum']} 击鼓",
            font=FONTS["body"],
            bg=COLORS["status_online"],
            fg=COLORS["bg_dark"],
            command=self._on_drum,
            relief="flat",
            cursor="hand2"
        ).pack(side="left", padx=2, expand=True, fill="x")

        tk.Button(
            btn_frame,
            text=f"{ICONS['bell']} 鸣金",
            font=FONTS["body"],
            bg=COLORS["status_offline"],
            fg=COLORS["bg_dark"],
            command=self._on_bell,
            relief="flat",
            cursor="hand2"
        ).pack(side="left", padx=2, expand=True, fill="x")

        return frame

    def _create_center_panel(self, parent) -> tk.Frame:
        """创建中间面板 - 战役态势"""
        frame = tk.Frame(parent, bg=COLORS["bg_dark"])

        # 战役态势面板
        self.battle_panel = BattleStatusPanel(frame)
        self.battle_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # 战术建议区
        tactics_frame = StyledFrame(frame, title="孙子兵法 · 战术建议")
        tactics_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.tactics_text = tk.Text(
            tactics_frame,
            font=FONTS["body"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_primary"],
            relief="flat",
            height=8,
            wrap="word"
        )
        self.tactics_text.pack(fill="both", expand=True, padx=5, pady=5)

        return frame

    def _create_right_panel(self, parent) -> tk.Frame:
        """创建右侧面板 - 军情速递"""
        frame = tk.Frame(parent, bg=COLORS["bg_dark"])

        # 军情速递面板
        self.report_panel = ReportPanel(frame)
        self.report_panel.pack(fill="both", expand=True, padx=5, pady=5)

        return frame

    # ========== 公共API ==========

    def add_general(
        self,
        name: str,
        status: str = "offline",
        role: str = "",
        message: str = ""
    ):
        """
        添加或更新将领

        Args:
            name: 将领名称
            status: 状态 (online/busy/idle/offline)
            role: 角色
            message: 当前消息
        """
        self.generals[name] = {
            "status": status,
            "role": role,
            "message": message
        }
        self._refresh_generals()
        self._update_stats()
        self._log(f"将领 {name} 状态更新: {status}")

    def remove_general(self, name: str):
        """移除将领"""
        if name in self.generals:
            del self.generals[name]
            self._refresh_generals()
            self._update_stats()
            self._log(f"将领 {name} 已撤离")

    def add_report(
        self,
        title: str,
        content: str,
        report_type: str = "info"
    ):
        """
        添加军情报告

        Args:
            title: 报告标题
            content: 报告内容
            report_type: 报告类型 (info/warning/danger/success)
        """
        self.reports.append({
            "title": title,
            "content": content,
            "type": report_type,
            "timestamp": datetime.now()
        })
        self.report_panel.add_report(title, content, report_type)
        self._log(f"收到军情: {title}", level=report_type)

    def update_battle_status(
        self,
        own_strength: int,
        enemy_strength: int,
        strategy: str = ""
    ):
        """更新战役态势"""
        self.battle_panel.update(own_strength, enemy_strength, strategy)

    def add_tactics(self, tactics: str):
        """添加战术建议"""
        self.tactics_text.insert("end", f"\n• {tactics}")
        self.tactics_text.see("end")

    def clear_tactics(self):
        """清空战术建议"""
        self.tactics_text.delete("1.0", "end")

    def set_commander(self, commander: Commander):
        """设置Commander实例"""
        self.commander = commander

    def set_memory(self, memory: Memory):
        """设置Memory实例"""
        self.memory = memory

    def set_llm_provider(self, provider: Any):
        """
        设置 LLM Provider 实例（配置军师）

        Args:
            provider: LLM Provider 实例
        """
        self.llm_provider = provider
        self._log(f"🧠 军师已任命：{provider}")

    def set_bingfu_instance(self, instance: Any):
        """设置 BingFu 实例"""
        self.bingfu_instance = instance

    # ========== 内部方法 ==========

    def _refresh_generals(self):
        """刷新将领列表"""
        # 清除现有卡片
        for widget in self.generals_frame.winfo_children():
            widget.destroy()

        # 重新创建卡片
        for name, info in self.generals.items():
            card = GeneralCard(
                self.generals_frame,
                name=name,
                status=info["status"],
                role=info["role"],
                message=info["message"]
            )
            card.pack(fill="x", pady=3, padx=3)

    def _update_stats(self):
        """更新统计信息"""
        online = sum(1 for g in self.generals.values() if g["status"] == "online")
        busy = sum(1 for g in self.generals.values() if g["status"] == "busy")
        total = len(self.generals)
        completed_pct = int(busy / total * 100) if total > 0 else 0

        self.stats_bar.update(
            generals=f"{online}人",
            tasks=f"{total}个",
            completed=f"{completed_pct}%",
            running=f"{busy}个"
        )

    def _log(self, message: str, level: str = "info"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "info": COLORS["text_secondary"],
            "warning": COLORS["status_busy"],
            "danger": COLORS["status_offline"],
            "success": COLORS["status_online"]
        }
        color = colors.get(level, COLORS["text_secondary"])

        self.log_area.config(state="normal")
        self.log_area.insert("end", f"[{timestamp}] {message}\n", level)
        self.log_area.tag_config(level, foreground=color)
        self.log_area.see("end")
        self.log_area.config(state="disabled")

    def _handle_command(self, command: str):
        """处理用户命令 —— 支持斜杠命令 + 自然语言"""
        if not command.strip():
            return

        self._log(f"传达军令: {command}")

        # 添加到历史
        self.command_history.append(command)
        self.history_index = len(self.command_history)

        # 斜杠命令优先解析
        if command.startswith("/help"):
            self._show_help()
        elif command.startswith("/add "):
            self._cmd_add_general(command[5:])
        elif command.startswith("/remove "):
            self._cmd_remove_general(command[8:])
        elif command.startswith("/report "):
            parts = command[8:].split(" ", 1)
            if len(parts) == 2:
                self.add_report(parts[0], parts[1])
        elif command.startswith("/battle "):
            self._cmd_update_battle(command[8:])
        elif command.startswith("/match "):
            self._cmd_match(command[8:])
        elif command.startswith("/smart "):
            self._cmd_smart(command[7:])
        elif command == "/clear":
            self.log_area.config(state="normal")
            self.log_area.delete("1.0", "end")
            self.log_area.config(state="disabled")
        elif command.startswith("/"):
            self._log("未知指令，输入 /help 查看斜杠命令帮助")
        else:
            # 自然语言理解层
            self._handle_natural_language(command)

    def _handle_natural_language(self, text: str):
        """
        自然语言指令处理器

        优先使用 LLM 理解，无 LLM 时降级为关键词匹配。
        """
        # === LLM 模式 ===
        if self.llm_provider:
            self._handle_with_llm(text)
            return

        # === 关键词匹配模式（降级方案） ===
        self._handle_with_keywords(text)

    def _handle_with_llm(self, text: str):
        """
        LLM 驱动的自然语言理解

        将用户输入和当前框架状态交给 LLM，
        LLM 返回结构化的操作指令，控制台执行。
        """
        self._log("🧠 军师正在理解军令...")

        # 构建上下文
        context = self._build_llm_context()

        system_prompt = (
            "你是兵符框架的中军帐AI调度官。用户用自然语言下达军令，"
            "你需要理解意图并返回JSON格式的操作指令。\n\n"
            "可用操作：\n"
            "1. {\"action\": \"query_generals\"} — 查询将领列表\n"
            "2. {\"action\": \"query_status\"} — 查询战役态势\n"
            "3. {\"action\": \"query_reports\"} — 查询军情报告\n"
            "4. {\"action\": \"add_general\", \"name\": \"...\", \"status\": \"online\", \"role\": \"...\"} — 添加将领\n"
            "5. {\"action\": \"remove_general\", \"name\": \"...\"} — 移除将领\n"
            "6. {\"action\": \"drum_all\"} — 全军出击\n"
            "7. {\"action\": \"assign_task\", \"agent_name\": \"...\", \"task\": \"...\"} — 给指定将领分配任务\n"
            "8. {\"action\": \"update_battle\", \"own\": N, \"enemy\": N, \"strategy\": \"...\"} — 更新战役态势\n"
            "9. {\"action\": \"clear_log\"} — 清空日志\n"
            "10. {\"action\": \"help\"} — 显示帮助\n"
            "11. {\"action\": \"chat\", \"message\": \"...\"} — 自由对话（无法匹配具体操作时使用）\n\n"
            "当前框架状态：\n" + context + "\n\n"
            "请只返回一个JSON对象，不要其他文字。"
        )

        # 在新线程中调用 LLM（避免阻塞 UI）
        def _llm_call():
            try:
                from bingfu.llm.base import LLMMessage, RoleType
                messages = [
                    LLMMessage(role=RoleType.SYSTEM, content=system_prompt),
                    LLMMessage(role=RoleType.USER, content=text),
                ]
                response = self.llm_provider.generate(messages, temperature=0.3, max_tokens=2048)
                content = response.content or ""

                # 解析 JSON
                # 尝试提取 JSON 部分
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    action = json.loads(json_match.group())
                    self._root.after(0, lambda: self._execute_llm_action(action, text))
                else:
                    self._root.after(0, lambda: self._log(f"💬 {content}"))

            except json.JSONDecodeError:
                self._root.after(0, lambda: self._log(f"⚠️ 军师回复格式异常，原始回复：{content[:100]}"))
            except Exception as e:
                self._root.after(0, lambda: self._log(f"❌ 军师调用失败：{e}"))
                # 降级到关键词匹配
                self._root.after(0, lambda: self._handle_with_keywords(text))

        thread = threading.Thread(target=_llm_call, daemon=True)
        thread.start()

    def _build_llm_context(self) -> str:
        """构建给 LLM 的框架状态上下文"""
        lines = []

        # 将领信息
        if self.generals:
            lines.append("将领列表：")
            for name, info in self.generals.items():
                lines.append(f"  - {name} ({info.get('status', 'unknown')}) {info.get('role', '')}")
        else:
            lines.append("暂无将领")

        # 战役态势
        try:
            own_txt = self.battle_panel.own_label.cget("text")
            enemy_txt = self.battle_panel.enemy_label.cget("text")
            lines.append(f"战役态势：己方 {own_txt} / 敌方 {enemy_txt}")
        except Exception:
            lines.append("暂无战役数据")

        # Agent 信息（如果有 BingFu 实例）
        if self.bingfu_instance and hasattr(self.bingfu_instance, 'agents'):
            if self.bingfu_instance.agents:
                lines.append("已注册 Agent：")
                for name, agent in self.bingfu_instance.agents.items():
                    has_llm = "🧠" if hasattr(agent, 'llm') and agent.llm else "⚠️"
                    profile_info = ""
                    if hasattr(agent, 'get_profile_summary'):
                        ps = agent.get_profile_summary()
                        if ps and ps != "无档案":
                            profile_info = f" | {ps}"
                    lines.append(f"  - {has_llm} {name} ({agent.role or '无角色'}){profile_info}")

        return "\n".join(lines)

    def _execute_llm_action(self, action: dict, original_text: str):
        """执行 LLM 返回的操作指令"""
        act = action.get("action", "chat")

        if act == "query_generals":
            total = len(self.generals)
            online = sum(1 for g in self.generals.values() if g["status"] == "online")
            busy = sum(1 for g in self.generals.values() if g["status"] == "busy")
            names = "、".join(self.generals.keys()) if self.generals else "（无）"
            self._log(f"📊 军中共有将领 {total} 位：{names}")
            self._log(f"   在线 {online} | 作战中 {busy} | 总计 {total}")

        elif act == "query_status":
            try:
                own_txt = self.battle_panel.own_label.cget("text")
                enemy_txt = self.battle_panel.enemy_label.cget("text")
                strategy = self.battle_panel.strategy_label.cget("text")
                self._log(f"⚔️ 当前战役态势：")
                self._log(f"   己方兵力：{own_txt} | 敌方兵力：{enemy_txt}")
                self._log(f"   战略建议：{strategy}")
            except Exception:
                self._log("⚔️ 暂无战役态势数据")

        elif act == "query_reports":
            if self.reports:
                self._log(f"📋 共收到 {len(self.reports)} 条军情：")
                for r in self.reports[-3:]:
                    self._log(f"   [{r.get('type', 'info').upper()}] {r['title']}：{r['content'][:40]}")
            else:
                self._log("📋 暂无军情报告")

        elif act == "add_general":
            name = action.get("name", "未名将领")
            status = action.get("status", "online")
            role = action.get("role", "")
            self.add_general(name, status, role)
            self._log(f"✅ 将领 {name} 已加入麾下")

        elif act == "remove_general":
            name = action.get("name", "")
            if name:
                self.remove_general(name)
                self._log(f"✅ 将领 {name} 已撤离")
            else:
                self._log("❌ 请指定将领姓名")

        elif act == "drum_all":
            self._on_drum()

        elif act == "assign_task":
            agent_name = action.get("agent_name", "")
            task = action.get("task", "")
            if not agent_name or not task:
                self._log("❌ 请指定将领名称和任务")
                return

            # 尝试在 BingFu 实例中找到 Agent 并执行
            if self.bingfu_instance and hasattr(self.bingfu_instance, 'agents'):
                agent = self.bingfu_instance.agents.get(agent_name)
                if agent:
                    self._log(f"🥁 向 {agent_name} 传达军令：{task}")
                    if hasattr(agent, 'llm') and agent.llm:
                        # 在新线程执行，避免阻塞 UI
                        def _exec():
                            try:
                                result = agent.drum(task)
                                # 分段输出完整结果（每段不超过500字符，自动换行）
                                def _show_result(res):
                                    for i in range(0, len(res), 500):
                                        chunk = res[i:i+500]
                                        self._root.after(0, lambda c=chunk: self._log(f"📋 {agent_name} 回报：\n{c}"))
                                _show_result(result)
                            except Exception as e:
                                self._root.after(0, lambda: self._log(f"❌ {agent_name} 执行失败：{e}"))
                        threading.Thread(target=_exec, daemon=True).start()
                    else:
                        agent.is_active = True
                        if agent_name in self.generals:
                            self.generals[agent_name]["status"] = "busy"
                            self._refresh_generals()
                            self._update_stats()
                        self._log(f"⚠️ {agent_name} 无 LLM 绑定，已标记为出战但无法智能执行")
                else:
                    self._log(f"❌ 未找到将领 {agent_name}")
            else:
                # 简单模式：更新 UI 状态
                if agent_name in self.generals:
                    self.generals[agent_name]["status"] = "busy"
                    self.generals[agent_name]["message"] = task
                    self._refresh_generals()
                    self._update_stats()
                    self._log(f"🥁 已向 {agent_name} 传达军令：{task}")
                else:
                    self._log(f"❌ 未找到将领 {agent_name}，可用将领：{', '.join(self.generals.keys()) or '（无）'}")

        elif act == "update_battle":
            own = action.get("own", 0)
            enemy = action.get("enemy", 0)
            strategy = action.get("strategy", "形势更新中")
            self.update_battle_status(own, enemy, strategy)
            self._log(f"✅ 战役态势已更新：己方 {own} | 敌方 {enemy}")

        elif act == "clear_log":
            self.log_area.config(state="normal")
            self.log_area.delete("1.0", "end")
            self.log_area.config(state="disabled")
            self._log("日志已清空")

        elif act == "help":
            self._show_help_natural()

        elif act == "smart_assign":
            # 智能派兵 — 自动评估任务并选择最优将领
            task = action.get("task", original_text)
            if not task:
                self._log("❌ 请提供任务描述")
                return
            self._log(f"🧮 正在分析任务并点兵：{task[:50]}...")
            # 在新线程执行
            def _smart():
                try:
                    if self.bingfu_instance and hasattr(self.bingfu_instance, 'smart_drum'):
                        result = self.bingfu_instance.smart_drum(task)
                        self._root.after(0, lambda: self._log(f"\n{result}"))
                    else:
                        self._root.after(0, lambda: self._log("❌ 框架不支持智能派兵"))
                except Exception as e:
                    self._root.after(0, lambda: self._log(f"❌ 智能派兵失败: {e}"))
            threading.Thread(target=_smart, daemon=True).start()

        elif act == "chat":
            # 自由对话，用 LLM 回复
            message = action.get("message", original_text)
            if self.llm_provider:
                def _chat():
                    try:
                        from bingfu.llm.base import LLMMessage, RoleType
                        msgs = [
                            LLMMessage(role=RoleType.SYSTEM, content="你是兵符框架的军师，用古代军事风格简短回复。"),
                            LLMMessage(role=RoleType.USER, content=message),
                        ]
                        resp = self.llm_provider.generate(msgs, temperature=0.8, max_tokens=1024)
                        content = resp.content or ""
                        # 完整输出，支持分段
                        def _show_chat(res):
                            for i in range(0, len(res), 500):
                                chunk = res[i:i+500]
                                prefix = "💬 军师曰：" if i == 0 else "    "
                                self._root.after(0, lambda c=chunk, p=prefix: self._log(f"{p}{c}"))
                        _show_chat(content)
                    except Exception as e:
                        self._root.after(0, lambda: self._log(f"❌ 军师回复失败：{e}"))
                threading.Thread(target=_chat, daemon=True).start()
            else:
                self._log(f"❓ 未能理解：「{message}」")

        else:
            self._log(f"❓ 未知操作类型：{act}")

    def _handle_with_keywords(self, text: str):
        """
        关键词匹配模式（无 LLM 时的降级方案）
        """
        t = text.lower()

        # ── 查询类 ──────────────────────────────────────────────
        # 查询将领数量
        if any(kw in t for kw in ["将士数量", "将领数量", "几位将领", "多少将领",
                                   "有哪些将领", "将领名单", "点兵", "兵力情况"]):
            total = len(self.generals)
            online = sum(1 for g in self.generals.values() if g["status"] == "online")
            busy   = sum(1 for g in self.generals.values() if g["status"] == "busy")
            idle   = sum(1 for g in self.generals.values() if g["status"] == "idle")
            offline = sum(1 for g in self.generals.values() if g["status"] == "offline")
            names = "、".join(self.generals.keys()) if self.generals else "（无）"
            self._log(f"📊 军中共有将领 {total} 位：{names}")
            self._log(f"   在线 {online} | 作战中 {busy} | 待命 {idle} | 离线 {offline}")
            return

        # 查询战况/当前态势
        if any(kw in t for kw in ["战况", "战情", "态势", "敌情", "战场形势",
                                   "目前情况", "当前情况", "现在怎么样", "形势"]):
            try:
                own_txt   = self.battle_panel.own_label.cget("text")
                enemy_txt = self.battle_panel.enemy_label.cget("text")
                strategy  = self.battle_panel.strategy_label.cget("text")
                self._log(f"⚔️  当前战役态势：")
                self._log(f"   己方兵力：{own_txt} | 敌方兵力：{enemy_txt}")
                self._log(f"   战略建议：{strategy}")
            except Exception:
                self._log("⚔️  暂无战役态势数据")
            return

        # 查询军情报告
        if any(kw in t for kw in ["军情", "情报", "报告", "有何军情", "战报"]):
            if self.reports:
                self._log(f"📋 共收到 {len(self.reports)} 条军情：")
                for r in self.reports[-3:]:  # 显示最近3条
                    self._log(f"   [{r.get('type','info').upper()}] {r['title']}：{r['content'][:40]}")
            else:
                self._log("📋 暂无军情报告")
            return

        # ── 操作类 ──────────────────────────────────────────────
        # 添加将领（识别格式：添加将领 <名字> [状态] [角色]）
        if any(kw in t for kw in ["添加将领", "加入将领", "新增将领", "招募将领", "部署将领"]):
            # 提取名字（取命令中的最后一个"词"，或提示用斜杠命令）
            words = text.replace("添加将领", "").replace("加入将领", "") \
                        .replace("新增将领", "").replace("招募将领", "") \
                        .replace("部署将领", "").strip().split()
            if words:
                name = words[0]
                status = words[1] if len(words) > 1 else "online"
                role   = words[2] if len(words) > 2 else ""
                self.add_general(name, status, role)
                self._log(f"✅ 将领 {name} 已加入麾下")
            else:
                self._log("请指定将领姓名，例如：添加将领 张辽 online 前锋")
            return

        # 移除/撤退将领
        if any(kw in t for kw in ["移除将领", "撤退将领", "撤销将领", "删除将领", "将领撤离"]):
            words = text.replace("移除将领", "").replace("撤退将领", "") \
                        .replace("撤销将领", "").replace("删除将领", "") \
                        .replace("将领撤离", "").strip().split()
            if words:
                name = words[0]
                self.remove_general(name)
                self._log(f"✅ 将领 {name} 已撤离")
            else:
                self._log("请指定将领姓名，例如：移除将领 张辽")
            return

        # 出击/击鼓
        if any(kw in t for kw in ["出击", "击鼓", "全军出击", "进攻", "冲锋", "发动进攻"]):
            self._on_drum()
            return

        # 收兵/鸣金
        if any(kw in t for kw in ["收兵", "鸣金", "撤退", "退兵", "鸣金收兵"]):
            self._log("提示：收兵请点击界面左下角「鸣金」按钮，需确认操作")
            return

        # 清空日志
        if any(kw in t for kw in ["清空日志", "清除日志", "清空记录", "清屏"]):
            self.log_area.config(state="normal")
            self.log_area.delete("1.0", "end")
            self.log_area.config(state="disabled")
            self._log("日志已清空")
            return

        # 更新战役态势（识别数字）
        import re
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2 and any(kw in t for kw in ["兵力", "对阵", "我军", "敌军", "更新战役", "更新态势"]):
            own, enemy = int(nums[0]), int(nums[1])
            strategy = "形势更新中"
            # 提取策略描述（引号内容 或 最后的词语）
            quote = re.findall(r'[「"](.*?)[」"]', text)
            if quote:
                strategy = quote[0]
            self.update_battle_status(own, enemy, strategy)
            self._log(f"✅ 战役态势已更新：己方 {own} | 敌方 {enemy}")
            return

        # 帮助
        if any(kw in t for kw in ["帮助", "help", "怎么用", "命令列表", "有什么命令"]):
            self._show_help_natural()
            return

        # ── 未能识别 ──────────────────────────────────────────
        self._log(f"❓ 未能识别指令「{text}」")
        self._log("   支持自然语言，如：战况如何 / 统计将士数量 / 全军出击")
        self._log("   或输入 /help 查看斜杠命令，输入「帮助」查看自然语言用法")

    def _cmd_add_general(self, args: str):
        """添加将领命令"""
        parts = args.split(" ", 2)
        name = parts[0]
        status = parts[1] if len(parts) > 1 else "offline"
        role = parts[2] if len(parts) > 2 else ""
        self.add_general(name, status, role)

    def _cmd_remove_general(self, name: str):
        """移除将领命令"""
        self.remove_general(name)

    def _cmd_update_battle(self, args: str):
        """更新战役命令"""
        parts = args.split()
        if len(parts) >= 2:
            try:
                own = int(parts[0])
                enemy = int(parts[1])
                strategy = parts[2] if len(parts) > 2 else ""
                self.update_battle_status(own, enemy, strategy)
            except ValueError:
                self._log("格式错误: /battle 己方兵力 敌方兵力 [策略]")

    def _cmd_match(self, task: str):
        """点兵命令 — 展示所有将军的匹配评分"""
        if not task.strip():
            self._log("用法: /match <任务描述>")
            return
        self._log(f"🧮 正在分析任务：{task[:50]}...")
        # 在新线程执行
        def _do_match():
            try:
                if self.bingfu_instance and hasattr(self.bingfu_instance, 'match_task'):
                    results = self.bingfu_instance.match_task(task)
                    if not results:
                        self._root.after(0, lambda: self._log("❌ 无可用将领"))
                        return
                    lines = ["\n📊 点兵结果：", "─" * 50]
                    for r in results:
                        lines.append(f"  {r.recommendation()}")
                    lines.append("─" * 50)
                    self._root.after(0, lambda: self._log("\n".join(lines)))
                else:
                    self._root.after(0, lambda: self._log("❌ 框架不支持点兵功能"))
            except Exception as e:
                self._root.after(0, lambda: self._log(f"❌ 点兵失败: {e}"))
        threading.Thread(target=_do_match, daemon=True).start()

    def _cmd_smart(self, task: str):
        """智能派兵命令 — 自动选最优将领执行"""
        if not task.strip():
            self._log("用法: /smart <任务描述>")
            return
        self._log(f"🥁 智能派兵中：{task[:50]}...")
        def _do_smart():
            try:
                if self.bingfu_instance and hasattr(self.bingfu_instance, 'smart_drum'):
                    result = self.bingfu_instance.smart_drum(task)
                    self._root.after(0, lambda: self._log(f"\n{result}"))
                else:
                    self._root.after(0, lambda: self._log("❌ 框架不支持智能派兵"))
            except Exception as e:
                self._root.after(0, lambda: self._log(f"❌ 智能派兵失败: {e}"))
        threading.Thread(target=_do_smart, daemon=True).start()

    def _show_help(self):
        """显示帮助"""
        help_text = """
╔════════════════════════════════════════════════════╗
║  兵符 · 中军帐 命令帮助                               ║
╠════════════════════════════════════════════════════╣
║  /add <将领> [状态] [角色]                           ║
║      添加将领 (状态: online/busy/idle/offline)        ║
║                                                     ║
║  /remove <将领>                                      ║
║      移除将领                                        ║
║                                                     ║
║  /report <标题> <内容>                               ║
║      添加军情报告                                    ║
║                                                     ║
║  /battle <己方> <敌方> [策略]                        ║
║      更新战役态势                                    ║
║                                                     ║
║  /match <任务描述>                                    ║
║      点兵 — 评估任务并展示所有将领匹配评分             ║
║                                                     ║
║  /smart <任务描述>                                    ║
║      智能派兵 — 自动选最优将领执行任务                ║
║                                                     ║
║  /clear                                              ║
║      清空日志                                        ║
╚════════════════════════════════════════════════════╝
"""
        self.log_area.config(state="normal")
        self.log_area.insert("end", help_text)
        self.log_area.see("end")
        self.log_area.config(state="disabled")

    def _show_help_natural(self):
        """显示自然语言帮助"""
        help_text = """
╔══════════════════════════════════════════════════════╗
║  兵符 · 中军帐 自然语言指令说明                         ║
╠══════════════════════════════════════════════════════╣
║  【查询类】                                           ║
║  统计我军将士数量 / 有哪些将领 / 点兵                     ║
║  目前战况如何 / 当前态势 / 战场形势                        ║
║  有何军情 / 战报 / 情报                               ║
║                                                     ║
║  【操作类】                                           ║
║  添加将领 张辽 online 前锋                             ║
║  移除将领 张辽                                        ║
║  全军出击 / 出击 / 进攻                               ║
║  清空日志 / 清屏                                      ║
║  更新战役 我军30000 敌军80000 「以逸待劳」               ║
║                                                     ║
║  【智能派兵】                                         ║
║  /match 分析这份数据报告                               ║
║  /smart 设计一个用户登录系统                           ║
║  让系统自动选择合适的将领（LLM理解后自动调用smart_assign）║
║                                                     ║
║  【斜杠命令】输入 /help 查看                           ║
╚══════════════════════════════════════════════════════╝
"""
        self.log_area.config(state="normal")
        self.log_area.insert("end", help_text)
        self.log_area.see("end")
        self.log_area.config(state="disabled")

    def _on_drum(self):
        """击鼓 - 全军出击"""
        self._log("🥁 击鼓！全军出击！")
        for name, info in self.generals.items():
            if info["status"] in ("online", "idle"):
                info["status"] = "busy"
        self._refresh_generals()
        self._update_stats()

    def _on_bell(self):
        """鸣金 - 收兵"""
        result = messagebox.askyesno("鸣金收兵", "确定要鸣金收兵吗？")
        if result:
            self._log("🔔 鸣金！收兵回营！")
            for name, info in self.generals.items():
                info["status"] = "offline"
            self._refresh_generals()
            self._update_stats()

    def _on_close(self):
        """窗口关闭处理"""
        result = messagebox.askyesno("退出", "确定要关闭中军帐吗？")
        if result:
            self.is_running = False
            self._root.destroy()

    def run(self, blocking: bool = True):
        """
        运行控制台

        Args:
            blocking: 是否阻塞运行
        """
        self.is_running = True
        self._log("兵符 · 中军帐 已启动")

        if self.llm_provider:
            self._log(f"🧠 军师已就位：{self.llm_provider}")
            self._log("支持自然语言智能指令，如：让韩信分析战况 / 给白起派个任务 / 战局如何")
        else:
            self._log("⚠️ 未配置军师(LLM)，自然语言使用关键词匹配模式")
            self._log("支持自然语言指令，如：战况如何 / 统计将士数量 / 全军出击")

        self._log("输入「帮助」查看自然语言用法，或输入 /help 查看斜杠命令")

        # 初始化示例数据
        self._init_demo_data()

        if blocking:
            self._root.mainloop()
        else:
            # 非阻塞模式 - 在新线程中运行
            thread = threading.Thread(target=self._root.mainloop, daemon=True)
            thread.start()

    def _init_demo_data(self):
        """初始化演示数据"""
        # 添加示例将领
        self.add_general("韩信", "online", "统帅", "正在分析战场形势")
        self.add_general("白起", "busy", "主将", "率领先锋部队突击")
        self.add_general("项羽", "idle", "虎将", "待命准备冲锋")
        self.add_general("诸葛亮", "online", "军师", "夜观星象推演战局")

        # 添加示例军情
        self.add_report("侦察回报", "敌军粮草队已过乌江，预计明日午时抵达", "success")
        self.add_report("斥候急报", "发现敌军增援部队约三万人", "warning")
        self.add_report("后勤报告", "我军粮草尚可支撑七日", "info")

        # 更新战役态势
        self.update_battle_status(30000, 80000, "敌众我寡，宜用奇兵")

        # 添加战术建议
        self.clear_tactics()
        self.tactics_text.insert("1.0", "• 以逸待劳，后发制人\n")
        self.tactics_text.insert("end", "• 诱敌深入，设伏聚歼\n")
        self.tactics_text.insert("end", "• 断其粮道，围而不攻")

    def stop(self):
        """停止控制台"""
        self.is_running = False
        if self._root:
            self._root.quit()


# 快捷函数
def create_console(**kwargs) -> MilitaryCommandConsole:
    """创建并返回控制台实例"""
    return MilitaryCommandConsole(**kwargs)


def launch_demo():
    """启动演示模式"""
    console = MilitaryCommandConsole()
    console.run()
