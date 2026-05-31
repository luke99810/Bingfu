"""
Tkinter 组件库
提供古代军事风格的可视化组件
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Dict, Any, List
from .styles import COLORS, FONTS, ICONS


class StyledFrame(tk.Frame):
    """带金色边框的样式化框架"""

    def __init__(self, parent, title: str = "", **kwargs):
        # 默认样式
        default_config = {
            "bg": COLORS["bg_medium"],
            "relief": "groove",
            "bd": 2,
            "highlightbackground": COLORS["gold_dark"],
        }
        default_config.update(kwargs)

        super().__init__(parent, **default_config)

        # 标题栏
        if title:
            self.title_frame = tk.Frame(self, bg=COLORS["gold_dark"], height=30)
            self.title_frame.pack(fill="x", padx=2, pady=(2, 0))
            self.title_frame.pack_propagate(False)

            self.title_label = tk.Label(
                self.title_frame,
                text=f"  {title}  ",
                font=FONTS["subtitle"],
                bg=COLORS["gold_dark"],
                fg=COLORS["bg_dark"],
                anchor="w"
            )
            self.title_label.pack(fill="x", padx=5, pady=5)


class GeneralCard(tk.Frame):
    """将领卡片组件"""

    def __init__(
        self,
        parent,
        name: str,
        status: str = "offline",
        role: str = "",
        message: str = "",
        **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.name = name
        self.status = status
        self.role = role
        self.message = message

        self._build_ui()

    def _build_ui(self):
        """构建卡片UI"""
        self.configure(
            bg=COLORS["bg_light"],
            relief="groove",
            bd=1,
            highlightbackground=COLORS["gold_dim"]
        )

        # 状态指示器
        status_colors = {
            "online": COLORS["status_online"],
            "busy": COLORS["status_busy"],
            "idle": COLORS["status_idle"],
            "offline": COLORS["status_offline"]
        }
        status_color = status_colors.get(self.status, COLORS["status_offline"])

        # 左侧状态条
        self.status_bar = tk.Frame(self, bg=status_color, width=4)
        self.status_bar.pack(side="left", fill="y")

        # 内容区
        content = tk.Frame(self, bg=COLORS["bg_light"])
        content.pack(side="left", fill="both", expand=True, padx=8, pady=5)

        # 名字和角色
        header = tk.Frame(content, bg=COLORS["bg_light"])
        header.pack(fill="x")

        name_label = tk.Label(
            header,
            text=f"{ICONS['star']} {self.name}",
            font=FONTS["body"],
            bg=COLORS["bg_light"],
            fg=COLORS["gold"]
        )
        name_label.pack(side="left")

        if self.role:
            role_label = tk.Label(
                header,
                text=f" [{self.role}]",
                font=FONTS["small"],
                bg=COLORS["bg_light"],
                fg=COLORS["text_secondary"]
            )
            role_label.pack(side="left")

        # 状态文字
        status_texts = {
            "online": "在线",
            "busy": "作战中",
            "idle": "待命",
            "offline": "离线"
        }
        status_label = tk.Label(
            header,
            text=status_texts.get(self.status, "未知"),
            font=FONTS["small"],
            bg=COLORS["bg_light"],
            fg=status_color
        )
        status_label.pack(side="right")

        # 当前消息
        if self.message:
            msg_label = tk.Label(
                content,
                text=self.message[:60] + ("..." if len(self.message) > 60 else ""),
                font=FONTS["small"],
                bg=COLORS["bg_light"],
                fg=COLORS["text_muted"],
                wraplength=200
            )
            msg_label.pack(anchor="w", pady=(5, 0))

    def update_status(self, status: str, message: str = ""):
        """更新状态"""
        self.status = status
        if message:
            self.message = message
        self._build_ui()


class BattleStatusPanel(tk.Frame):
    """战役态势面板"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._build_ui()

    def _build_ui(self):
        """构建面板UI"""
        self.configure(bg=COLORS["bg_dark"])

        # 标题
        title = tk.Label(
            self,
            text=f"{ICONS['battle']} 战役态势",
            font=FONTS["subtitle"],
            bg=COLORS["bg_dark"],
            fg=COLORS["gold"]
        )
        title.pack(pady=(10, 5))

        # 双方兵力显示
        info_frame = tk.Frame(self, bg=COLORS["bg_dark"])
        info_frame.pack(fill="x", padx=20, pady=10)

        # 己方
        self.own_frame = tk.Frame(info_frame, bg=COLORS["bg_dark"])
        self.own_frame.pack(side="left", fill="both", expand=True)

        tk.Label(
            self.own_frame,
            text="己方兵力",
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_secondary"]
        ).pack()

        self.own_label = tk.Label(
            self.own_frame,
            text="30000",
            font=FONTS["title"],
            bg=COLORS["bg_dark"],
            fg=COLORS["status_online"]
        )
        self.own_label.pack()

        # 中间箭头
        arrow_label = tk.Label(
            info_frame,
            text=f"{ICONS['arrow']} VS {ICONS['arrow']}",
            font=FONTS["subtitle"],
            bg=COLORS["bg_dark"],
            fg=COLORS["gold"]
        )
        arrow_label.pack(side="left", padx=15)

        # 敌方
        self.enemy_frame = tk.Frame(info_frame, bg=COLORS["bg_dark"])
        self.enemy_frame.pack(side="right", fill="both", expand=True)

        tk.Label(
            self.enemy_frame,
            text="敌方兵力",
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_secondary"]
        ).pack()

        self.enemy_label = tk.Label(
            self.enemy_frame,
            text="80000",
            font=FONTS["title"],
            bg=COLORS["bg_dark"],
            fg=COLORS["status_offline"]
        )
        self.enemy_label.pack()

        # 策略建议
        strategy_frame = tk.Frame(self, bg=COLORS["bg_medium"], padx=10, pady=10)
        strategy_frame.pack(fill="x", padx=15, pady=(0, 10))

        tk.Label(
            strategy_frame,
            text="战略建议",
            font=FONTS["body"],
            bg=COLORS["bg_medium"],
            fg=COLORS["gold"]
        ).pack(anchor="w")

        self.strategy_label = tk.Label(
            strategy_frame,
            text="以逸待劳，后发制人",
            font=FONTS["body"],
            bg=COLORS["bg_medium"],
            fg=COLORS["text_primary"],
            wraplength=250,
            justify="left"
        )
        self.strategy_label.pack(anchor="w")

    def update(self, own: int, enemy: int, strategy: str):
        """更新态势数据"""
        self.own_label.config(text=str(own))
        self.enemy_label.config(text=str(enemy))
        self.strategy_label.config(text=strategy)


class ReportPanel(tk.Frame):
    """军情速递面板"""

    def __init__(self, parent, max_reports: int = 10, **kwargs):
        super().__init__(parent, **kwargs)
        self.max_reports = max_reports
        self.reports: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self):
        """构建面板UI"""
        self.configure(bg=COLORS["bg_medium"])

        # 标题栏
        header = tk.Frame(self, bg=COLORS["gold_dark"], height=30)
        header.pack(fill="x", padx=2, pady=(2, 0))
        header.pack_propagate(False)

        tk.Label(
            header,
            text=f"  {ICONS['report']} 军情速递  ",
            font=FONTS["subtitle"],
            bg=COLORS["gold_dark"],
            fg=COLORS["bg_dark"]
        ).pack(side="left")

        # 清空按钮
        clear_btn = tk.Button(
            header,
            text="清空",
            font=FONTS["small"],
            bg=COLORS["gold_dark"],
            fg=COLORS["bg_dark"],
            command=self.clear,
            relief="flat",
            cursor="hand2"
        )
        clear_btn.pack(side="right", padx=5, pady=2)

        # 报告列表容器
        self.list_frame = tk.Frame(self, bg=COLORS["bg_medium"])
        self.list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 滚动条
        scrollbar = tk.Scrollbar(self.list_frame)
        scrollbar.pack(side="right", fill="y")

        self.canvas = tk.Canvas(
            self.list_frame,
            bg=COLORS["bg_medium"],
            highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.canvas.yview)

        self.content_frame = tk.Frame(self.canvas, bg=COLORS["bg_medium"])
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def add_report(self, title: str, content: str, report_type: str = "info"):
        """添加报告"""
        colors = {
            "info": COLORS["status_idle"],
            "warning": COLORS["status_busy"],
            "danger": COLORS["status_offline"],
            "success": COLORS["status_online"]
        }
        color = colors.get(report_type, COLORS["text_secondary"])

        report = tk.Frame(self.content_frame, bg=COLORS["bg_light"], padx=8, pady=5)
        report.pack(fill="x", pady=2)

        # 类型图标
        icons = {
            "info": "📋",
            "warning": "⚠️",
            "danger": "🔴",
            "success": "✅"
        }

        tk.Label(
            report,
            text=f"{icons.get(report_type, '📋')} {title}",
            font=FONTS["body"],
            bg=COLORS["bg_light"],
            fg=color
        ).pack(anchor="w")

        tk.Label(
            report,
            text=content[:100] + ("..." if len(content) > 100 else ""),
            font=FONTS["small"],
            bg=COLORS["bg_light"],
            fg=COLORS["text_muted"]
        ).pack(anchor="w")

        # 添加到列表
        self.reports.insert(0, {"title": title, "content": content, "type": report_type})

        # 限制数量
        if len(self.reports) > self.max_reports:
            self.reports.pop()
            for child in self.content_frame.winfo_children():
                child.destroy()
            for report_data in self.reports:
                self.add_report_widget(report_data)

    def add_report_widget(self, report_data: Dict):
        """添加报告组件"""
        colors = {
            "info": COLORS["status_idle"],
            "warning": COLORS["status_busy"],
            "danger": COLORS["status_offline"],
            "success": COLORS["status_online"]
        }
        color = colors.get(report_data["type"], COLORS["text_secondary"])

        icons = {
            "info": "📋",
            "warning": "⚠️",
            "danger": "🔴",
            "success": "✅"
        }

        report = tk.Frame(self.content_frame, bg=COLORS["bg_light"], padx=8, pady=5)
        report.pack(fill="x", pady=2)

        tk.Label(
            report,
            text=f"{icons.get(report_data['type'], '📋')} {report_data['title']}",
            font=FONTS["body"],
            bg=COLORS["bg_light"],
            fg=color
        ).pack(anchor="w")

        tk.Label(
            report,
            text=report_data["content"][:100] + ("..." if len(report_data["content"]) > 100 else ""),
            font=FONTS["small"],
            bg=COLORS["bg_light"],
            fg=COLORS["text_muted"]
        ).pack(anchor="w")

    def clear(self):
        """清空所有报告"""
        self.reports.clear()
        for child in self.content_frame.winfo_children():
            child.destroy()


class StatsBar(tk.Frame):
    """底部状态栏"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.stats = {}
        self._build_ui()

    def _build_ui(self):
        """构建状态栏UI"""
        self.configure(
            bg=COLORS["gold_dark"],
            height=45,
            relief="groove",
            bd=2
        )
        self.pack_propagate(False)

        self.inner_frame = tk.Frame(self, bg=COLORS["gold_dark"])
        self.inner_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # 默认统计项（增大间距，给数字足够显示空间）
        self._stat_labels = {}
        default_stats = [
            ("generals", "在线将领", "0"),
            ("tasks", "任务总数", "0"),
            ("completed", "完成率", "0%"),
            ("running", "进行中", "0")
        ]

        for key, label_text, default_value in default_stats:
            # 使用固定宽度 frame 防止数字被截断
            stat_frame = tk.Frame(self.inner_frame, bg=COLORS["gold_dark"], width=120)
            stat_frame.pack(side="left", padx=15, pady=2)
            stat_frame.pack_propagate(False)  # 保持固定宽度

            tk.Label(
                stat_frame,
                text=label_text,
                font=FONTS["small"],
                bg=COLORS["gold_dark"],
                fg=COLORS["bg_dark"],
                anchor="center",
                width=8
            ).pack()

            value_label = tk.Label(
                stat_frame,
                text=default_value,
                font=FONTS["subtitle"],
                bg=COLORS["gold_dark"],
                fg=COLORS["bg_dark"],
                anchor="center",
                width=8
            )
            value_label.pack()
            self._stat_labels[key] = value_label

    def update(self, **kwargs):
        """更新统计值"""
        for key, value in kwargs.items():
            if key in self._stat_labels:
                self._stat_labels[key].config(text=str(value))


class CommandInput(tk.Frame):
    """命令输入框"""

    def __init__(self, parent, on_submit: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_submit = on_submit
        self._build_ui()

    def _build_ui(self):
        """构建输入框UI"""
        self.configure(bg=COLORS["bg_dark"], bd=1, highlightbackground=COLORS["gold_dark"], highlightthickness=1)

        # 提示符
        prompt = tk.Label(
            self,
            text=f"{ICONS['dot']} 军令:",
            font=FONTS["mono"],
            bg=COLORS["bg_medium"],
            fg=COLORS["gold"]
        )
        prompt.pack(side="left", padx=(10, 5), pady=6)

        # 输入框
        self.entry = tk.Entry(
            self,
            font=FONTS["mono"],
            bg=COLORS["bg_light"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["gold"],
            relief="flat",
            bd=5
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=6)

        # 绑定回车事件
        self.entry.bind("<Return>", self._handle_submit)

        # 发送按钮
        send_btn = tk.Button(
            self,
            text="传达",
            font=FONTS["body"],
            bg=COLORS["gold"],
            fg=COLORS["bg_dark"],
            command=self._handle_submit,
            relief="flat",
            cursor="hand2",
            padx=15
        )
        send_btn.pack(side="right", padx=(0, 10), pady=6)

    def _handle_submit(self, event=None):
        """处理提交"""
        command = self.entry.get().strip()
        if command and self.on_submit:
            self.on_submit(command)
            self.entry.delete(0, tk.END)

    def get(self) -> str:
        """获取输入内容"""
        return self.entry.get()

    def clear(self):
        """清空输入"""
        self.entry.delete(0, tk.END)
