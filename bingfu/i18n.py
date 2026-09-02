r"""中英文切换。

════════════════════════════════════════════════════════════════
 为什么不是「界面翻译一下」那么简单
════════════════════════════════════════════════════════════════

这个框架的可见文字分布在三处，只改一处就等于没改：

  · 界面 chrome —— 面板标题、按钮、状态词
  · 战役事件 —— 「庙算完毕」「点将：各将匹配评分」这类阶段标题
  · 模型输出 —— 将领回复用什么语言，由系统提示决定

三处都切了才叫切换语言。只切界面的话，一个英文界面里蹦出整段中文
军令式回复，比不做还难看。

════════════════════════════════════════════════════════════════
 漏翻必须能被发现
════════════════════════════════════════════════════════════════

★ 缺词条时回退到中文，同时**记一笔**。

  回退到中文是为了不让界面出现 ``app.title`` 这种裸 key；
  但只回退不记录的话，漏翻就变成了一个安静的洞 ——
  英文界面里夹着几句中文，谁也说不清是漏了还是故意留的。
  ``missing_keys()`` 把它变成可以断言的东西，
  而 ``tests/test_i18n.py`` 里有一条测试要求 en 覆盖全部 key。

★ 术语保留军事叙事。

  General / Command Tent / Muster 之类，不译成 Agent / Dashboard。
  这套叙事是框架的设计主张，翻译时把它抹平就抹掉了主张本身。
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List

#: 支持的语言
LOCALES = ("zh", "en")

_lock = threading.Lock()
_current = "zh"
_missing: set = set()
#: 语言变化时要被通知的回调（界面用它重绘）
_observers: List[Callable[[str], None]] = []


# ══════════════════════════════════════════════════════════
#  词条
# ══════════════════════════════════════════════════════════

CATALOG: Dict[str, Dict[str, str]] = {
    # ── 应用 ──────────────────────────────────────────
    "app.title": {"zh": "兵符 · 中军帐", "en": "BingFu · Command Tent"},
    "app.subtitle": {"zh": "Multi-Agent 战役指挥系统",
                     "en": "Multi-Agent Campaign Command System"},
    "app.started": {"zh": "兵符 · 中军帐 已启动",
                    "en": "BingFu · Command Tent is up"},
    "app.lang_button": {"zh": "EN", "en": "中"},
    "app.lang_tip": {"zh": "切换为英文", "en": "Switch to Chinese"},
    "app.lang_switched": {"zh": "界面语言已切换为中文",
                          "en": "Interface language switched to English"},

    # ── 面板 ──────────────────────────────────────────
    "panel.generals": {"zh": "将领名录", "en": "Roster"},
    "panel.battle": {"zh": "战役态势", "en": "Battle Status"},
    "panel.reports": {"zh": "军情速递", "en": "Dispatches"},
    "panel.log": {"zh": "战报传回", "en": "Field Log"},
    "panel.tactics": {"zh": "孙子兵法 · 战术建议",
                      "en": "Art of War · Tactical Advice"},

    # ── 战役态势 ──────────────────────────────────────
    "battle.ours": {"zh": "己方兵力", "en": "Our Strength"},
    "battle.theirs": {"zh": "敌方兵力", "en": "Enemy Strength"},
    "battle.strategy": {"zh": "战略建议", "en": "Strategy"},
    "battle.no_order": {"zh": "尚未受领军令", "en": "No order received yet"},

    # ── 按钮与输入 ────────────────────────────────────
    "btn.drum": {"zh": "击鼓", "en": "Drum"},
    "btn.gong": {"zh": "鸣金", "en": "Gong"},
    "btn.clear": {"zh": "清空", "en": "Clear"},
    "btn.send": {"zh": "传达", "en": "Send"},
    "input.order": {"zh": "军令：", "en": "Order:"},

    # ── 将领状态 ──────────────────────────────────────
    "status.online": {"zh": "在线", "en": "Online"},
    "status.busy": {"zh": "作战中", "en": "Engaged"},
    "status.idle": {"zh": "待命", "en": "Standby"},
    "status.offline": {"zh": "离线", "en": "Offline"},
    "stats.tasks": {"zh": "任务总数", "en": "Tasks"},
    "stats.completed": {"zh": "完成率", "en": "Completed"},
    "stats.running": {"zh": "进行中", "en": "Running"},
    "log.booting": {"zh": "系统启动中...", "en": "Starting up..."},
    "log.status_update": {"zh": "将领 {name} 状态更新: {status}",
                          "en": "General {name} status -> {status}"},
    "log.dispatch_in": {"zh": "收到军情: {title}", "en": "Dispatch received: {title}"},
    "stats.generals": {"zh": "在线将领", "en": "Generals"},

    # ── 战役阶段（后端事件） ──────────────────────────
    "campaign.order": {"zh": "军令已受理", "en": "Order accepted"},
    "campaign.assess": {"zh": "庙算完毕，敌方战力已估",
                        "en": "Assessment done — enemy strength estimated"},
    "campaign.assess_failed": {"zh": "庙算失败", "en": "Assessment failed"},
    "campaign.muster": {"zh": "点将：各将匹配评分",
                        "en": "Muster — match scores for every general"},
    "campaign.muster_failed": {"zh": "点将失败", "en": "Muster failed"},
    "campaign.muster_score_failed": {
        "zh": "点将评分失败（不影响出兵）",
        "en": "Match scoring failed (the campaign proceeds anyway)"},
    "campaign.march": {"zh": "交兵：开始执行", "en": "Engage — execution begins"},
    "campaign.march_failed": {"zh": "执行失败", "en": "Execution failed"},
    "campaign.done": {"zh": "全军复命", "en": "All units reporting back"},
    "campaign.advise_assess": {"zh": "军师进言（庙算）",
                               "en": "Strategist's counsel (assessment)"},
    "campaign.advise_report": {"zh": "军师进言（复命）",
                               "en": "Strategist's counsel (debrief)"},
    "campaign.no_tools": {"zh": "注意：本次未调用任何工具",
                          "en": "Note: no tool was called in this run"},
    "campaign.plan_shape": {"zh": "拆解形状", "en": "Plan shape"},
    "campaign.plan_shape_detail": {
        "zh": "{n} 个子任务、{edges} 条依赖，并行宽度 {width}",
        "en": "{n} subtasks, {edges} edges, parallel width {width}"},
    "campaign.overwrite": {"zh": "重复覆盖", "en": "Repeated overwrite"},
    "campaign.plan_issue": {"zh": "拆解告警", "en": "Plan warning"},
    "campaign.checkpoint": {"zh": "断点", "en": "Checkpoint"},

    # ── 指令 ──────────────────────────────────────────
    "cmd.received": {"zh": "收到指令", "en": "Command received"},
    "cmd.unknown": {"zh": "未知指令，输入 /help 查看斜杠命令",
                    "en": "Unknown command — type /help for the list"},
    "cmd.war_usage": {"zh": "用法: /war <任务>   或   /war <任务> || <要求>",
                      "en": "Usage: /war <task>   or   /war <task> || <requirements>"},
    "cmd.no_framework": {"zh": "未接入框架实例，无法发兵",
                         "en": "No framework instance attached — cannot dispatch"},
    "cmd.order": {"zh": "军令", "en": "Order"},
    "cmd.requirements": {"zh": "要求", "en": "Requirements"},
    "cmd.match_usage": {"zh": "用法: /match <任务描述>",
                        "en": "Usage: /match <task description>"},
    "cmd.analysing": {"zh": "正在分析任务", "en": "Analysing the task"},
    "cmd.no_generals": {"zh": "无可用将领", "en": "No general available"},
    "cmd.muster_result": {"zh": "点兵结果：", "en": "Muster results:"},
    "cmd.lang_usage": {"zh": "用法: /lang zh   或   /lang en",
                       "en": "Usage: /lang zh   or   /lang en"},
    "cmd.lang_unknown": {"zh": "未知语言代码，只支持 zh 与 en",
                         "en": "Unknown locale — only zh and en are supported"},

    # ── 帮助 ──────────────────────────────────────────
    "help.title": {"zh": "兵符 · 中军帐 命令帮助",
                   "en": "BingFu · Command Tent — Command Help"},
    "help.add": {"zh": "添加将领 (状态: online/busy/idle/offline)",
                 "en": "Add a general (status: online/busy/idle/offline)"},
    "help.remove": {"zh": "移除将领", "en": "Remove a general"},
    "help.report": {"zh": "添加军情报告", "en": "Add a dispatch"},
    "help.battle": {"zh": "更新战役态势", "en": "Update the battle status"},
    "help.match": {"zh": "点兵 — 评估任务并展示所有将领匹配评分",
                   "en": "Muster — score every general against the task"},
    "help.smart": {"zh": "智能派兵 — 自动选最优将领执行任务",
                   "en": "Smart dispatch — pick the best general and execute"},
    "help.war": {"zh": "发号施令 — 走完整战役流程",
                 "en": "Issue an order — run the full campaign"},
    "help.war_flow": {"zh": "庙算(敌方战力) → 点将(我方战力) → 交兵 → 复命",
                      "en": "Assess → Muster → Engage → Debrief"},
    "help.clear": {"zh": "清空日志", "en": "Clear the log"},
    "help.lang": {"zh": "切换界面语言（中文 / English）",
                  "en": "Switch interface language (Chinese / English)"},

    # ── 系统就绪卡片 ──────────────────────────────────
    "boot.ready": {"zh": "系统就绪", "en": "System ready"},
    "boot.awaiting": {"zh": "待受军令", "en": "Awaiting orders"},
    "boot.awaiting_detail": {
        "zh": "输入 /war <任务> 下达军令，或 /help 查看全部指令",
        "en": "Type /war <task> to issue an order, or /help for all commands"},
    "boot.generals": {"zh": "位将领待命", "en": "generals on standby"},
    "boot.tools": {"zh": "兵器", "en": "tools"},
    "boot.workspace": {"zh": "工作区", "en": "workspace"},
    "boot.episodes": {"zh": "战报", "en": "episodes"},
    "boot.knowledge": {"zh": "知识", "en": "facts"},

    # ── 给模型的语言指示（后端） ──────────────────────
    "prompt.reply_language": {
        "zh": "用简练有力的军令风格回复，使用中文。",
        "en": "Reply in English, in the terse register of a military order."},
}


# ══════════════════════════════════════════════════════════
#  接口
# ══════════════════════════════════════════════════════════

def get_locale() -> str:
    return _current


def set_locale(locale: str) -> str:
    """切换语言并通知所有观察者。返回实际生效的语言。

    ★ 不认识的语言代码**不静默忽略** —— 抛 ValueError。
      静默保持原样的话，调用方会以为切过去了，
      而界面没变的原因永远查不出来。
    """

    global _current
    if locale not in LOCALES:
        raise ValueError("不支持的语言：%r（可选：%s）" % (locale, ", ".join(LOCALES)))
    with _lock:
        _current = locale
    for cb in list(_observers):
        try:
            cb(locale)
        except Exception:       # noqa: BLE001
            # ★ 某个界面组件重绘失败，不该让整次切换半途而废：
            #   那会留下一个「一半中文一半英文」的界面。
            pass
    return _current


def t(key: str, **fmt) -> str:
    """取词条。缺失时回退中文并记一笔。"""

    entry = CATALOG.get(key)
    if entry is None:
        _missing.add(key)
        return key
    text = entry.get(_current)
    if text is None:
        _missing.add("%s[%s]" % (key, _current))
        text = entry.get("zh", key)
    return text.format(**fmt) if fmt else text


def missing_keys() -> List[str]:
    """运行期遇到过的缺失词条。让漏翻可被断言，而不是安静地存在。"""

    return sorted(_missing)


def on_change(callback: Callable[[str], None]) -> Callable[[str], None]:
    """注册语言变化回调（界面用它重绘）。"""

    _observers.append(callback)
    return callback


def off_change(callback: Callable[[str], None]) -> None:
    if callback in _observers:
        _observers.remove(callback)


def reset_for_tests() -> None:
    """测试之间恢复默认，避免相互污染。"""

    global _current
    with _lock:
        _current = "zh"
    _missing.clear()
    _observers.clear()
