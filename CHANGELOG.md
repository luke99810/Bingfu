# Changelog (变更日志)

All notable changes to the **BingFu (兵符)** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] (未发布)

---

## [0.5.0] - 2026-05-19

### Added (新增)

- **将军战力体系** 🆕
  - `bingfu/profile.py` — 将军档案数据模型
    - `CombatStyle` 枚举：5种作战风格（谋略型/突击型/统帅型/勇武型/侦察型）
    - `CombatStats` 模型：五维战力值（攻击力/防御力/谋略值/速度/智力），1-100分
    - `GeneralProfile` 模型：将军完整档案（风格+专长+弱项+战力+描述）
  - `bingfu/assessment.py` — 军情评估模块
    - `TaskComplexity` 枚举：4级难度（易/中/难/极难）
    - `TaskAssessment` 模型：评估结果（复杂度/所需能力/敌方战力）
    - `TaskAssessor` 类：LLM优先+规则降级评估器
  - `bingfu/matcher.py` — 点兵台智能匹配引擎
    - `MatchResult` 模型：单将军匹配结果
    - `TaskMatcher` 类：加权评分算法（专长40%+战力30%+作风20%-弱项10%）
  - `bingfu/presets.py` — 名将预设档案
    - 韩信(统帅型)、白起(突击型)、诸葛亮(谋略型)、项羽(勇武型)
    - 斥候(侦察型)、谋士、猛将 三个通用预设
    - `get_preset(name)` 和 `list_presets()` 工具函数

- **智能派兵系统** 🆕
  - `BingFu.smart_drum(task)` — 自动选最优将领执行任务
  - `BingFu.match_task(task)` — 展示所有将领匹配评分
  - `BingFu.assess_task(task)` — 评估任务难度
  - `Commander.smart_drum(task)` — 指挥系统智能派兵
  - `Commander.match_task(task)` — 点兵接口
  - `Commander.assess_task(task)` — 军情评估接口
  - `Commander.drum_one/all()` — **修复桩方法**，实际调用 agent.drum()
  - `Commander.gong_one/all()` — **修复桩方法**，实际调用 agent.gong()
  - `BingFu.drum/gong()` — **修复桩方法**
  - `coordinate()` 新增 `"smart"` 策略

- **Agent 战力档案支持** 🆕
  - `Agent.profile` 字段：挂载 GeneralProfile
  - `Agent.get_profile_summary()` — 战力档案摘要
  - `execute()` 自动注入档案信息到 system_prompt

- **控制台智能派兵** 🆕
  - `/match <任务>` — 点兵命令，展示所有将领匹配评分
  - `/smart <任务>` — 智能派兵，自动选最优将领执行
  - LLM action `smart_assign` — 自然语言理解后自动调用智能派兵
  - `_build_llm_context()` 包含将领档案信息

- **启动脚本增强**
  - `launch.py` v0.5.0 — 自动加载预设档案，启用指挥系统

### Changed (变更)

- `bingfu/__init__.py` — 导出新增模块（profile/assessment/matcher/presets）
- `pyproject.toml` — 版本更新至 0.5.0，Python 版本要求降至 3.9
- `bingfu/bingfu.py` — 版本更新至 0.5.0
- 控制台标题更新为 v0.5.0

### Fixed (修复)

- `Commander.drum_one/all()` — 从占位字符串改为实际调用 agent.drum()
- `Commander.gong_one/all()` — 从占位字符串改为实际调用 agent.gong()
- `BingFu.drum()` — 从占位字符串改为实际调用 agent.drum()
- `BingFu.gong()` — 从占位字符串改为实际调用 agent.gong()
- `pydantic` 字段命名冲突：`__matcher` 替代 `_matcher`

---

## [0.4.0] - 2026-05-18

### Added (新增)

- **孙子兵法战术引擎**
  - `bingfu/tactics.py` — 完整的孙子兵法十三篇战术实现
    - `TacticsEngine` 类：战场态势分析
    - `SunTzuAgent` 类：智能军师Agent
    - `TacticType` 枚举：13种战术类型
    - `TacticalContext` 模型：战术上下文

- **古代名将示例**
  - `examples/famous_generals.py` — 十大名将Agent实现
    - 白起、韩信、项羽、诸葛亮、岳飞等
    - 完整战役模拟场景

- **工具与记忆示例**
  - `examples/tool_usage.py` — 工具使用完整示例
  - `examples/memory_usage.py` — 记忆系统完整示例

- **CLI指南**
  - `examples/cli_guide.py` — 完整CLI使用文档

- **单元测试补全**
  - `tests/test_tool.py` — Tool类完整测试
  - `tests/test_memory.py` — Memory类完整测试

### Changed (变更)
- 更新README，添加新功能说明
- 更新`__init__.py`，导出tactics模块
- 完善项目结构文档

---

## [0.3.0] - 2026-05-18

### Added (新增)

- **中军帐可视化模块** 🆕
  - `bingfu/visual/` — Tkinter桌面可视化组件
    - `styles.py` — 古代军事风格配色与字体常量
    - `components.py` — UI组件库（GeneralCard, BattleStatusPanel, ReportPanel等）
    - `console.py` — `MilitaryCommandConsole` 主控制台

- **可视化组件**
  - `GeneralCard` — 将领卡片（状态指示器、角色显示）
  - `BattleStatusPanel` — 战役态势面板（双方兵力、战略建议）
  - `ReportPanel` — 军情速递面板（报告列表、类型区分）
  - `StatsBar` — 底部状态栏（统计信息）
  - `CommandInput` — 命令输入框
  - `StyledFrame` — 金色边框样式框架

- **控制台功能**
  - 将领状态管理（添加/移除/更新）
  - 实时战役态势更新
  - 军情报告系统（info/warning/danger/success）
  - 命令行交互（/add, /remove, /report, /battle, /help, /clear）
  - 击鼓/鸣金快捷操作
  - 日志输出区域
  - 孙子兵法战术建议区

- **示例文件**
  - `examples/console_demo.py` — 中军帐演示（基础/集成/实时模式）

### Changed (变更)
- 更新README添加可视化模块文档
- 更新项目结构说明
- 更新开发状态（可视化已标记完成）

---

## [0.1.0] - 2026-05-18

### Added (新增)
- Initial project scaffolding (初始项目脚手架)
- Basic package structure (基础包结构)
- Placeholder for core modules (核心模块占位符)

---

## Template for future versions (未来版本模板)

## [X.Y.Z] - YYYY-MM-DD

### Added (新增)
- 

### Changed (变更)
- 

### Deprecated (弃用)
- 

### Removed (移除)
- 

### Fixed (修复)
- 

### Security (安全)
- 
