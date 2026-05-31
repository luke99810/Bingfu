# BingFu (兵符) — Lightweight Multi‑Agent Framework

[English](#english) | [中文](#中文)

---

# English

## 🎯 Introduction

**BingFu (兵符)** is a lightweight multi‑agent framework inspired by ancient Chinese warfare. It allows you to create **single agents** or **multi‑agent systems** that communicate using **war‑style signals**:

- **Drum (击鼓)** — start an agent or a mission
- **Gong (鸣金)** — stop an agent or abort a mission

The framework is designed to be simple, extensible, and fun, with a nod to Chinese military strategy.

## ✨ Features

- **Lightweight**: Minimal dependencies, fast to get started
- **Single & Multi‑Agent**: Support both solo agents and coordinated teams
- **Signal System**: Drum/Gong signals for clear control flow
- **Extensible**: Easily add custom agents, tools, and memory
- **Chinese Cultural Twist**: Naming and concepts from ancient Chinese warfare
- **Tactics Engine**: Built‑in Sun Tzu's Art of War strategy support
- **Famous Generals**: Pre‑built agents inspired by legendary Chinese generals
- **Visualization**: Desktop console (MilitaryCommandConsole) with ancient military style
- **LangChain Integration**: Support for LangChain agents, memory, and RAG retrieval

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/luke99810/bingfu.git
cd bingfu

# Install the package (editable mode)
pip install -e .
```

## 🚀 Quick Start

### 5-Minute Tutorial

Run the quickstart guide:

```bash
python examples/quickstart.py
```

This covers all core features in 7 simple examples.

### Basic Usage

```python
from bingfu import BingFu, Agent

# Create a single agent
agent = Agent(name="Scout", role="Reconnaissance")

# Create the BingFu master
master = BingFu(name="Commander")
master.add_agent(agent)

# Send drum signal (start)
result = master.drum("Scout", "Scout enemy camp")
print(result)

# Send gong signal (stop)
result = master.gong("Scout")
print(result)
```

### Multi-Agent Coordination

```python
from bingfu import BingFu, Agent

# Create BingFu master
master = BingFu(name="Battle Commander")

# Create multiple agents
agents = [
    Agent(name="Vanguard", role="Lead Attack"),
    Agent(name="Main Force", role="Main Assault"),
    Agent(name="Reserve", role="Backup"),
]

# Add all agents
for agent in agents:
    master.add_agent(agent)

# Enable commander mode
master.enable_commander(name="Marshal")

# Drum all agents (start all)
results = master.drum_all("Launch Attack")

# Gong all agents (stop all)
results = master.gong_all()
```

### Using Tools

```python
from bingfu import Agent, Tool

# Define a tool function
def scout_location(location: str) -> str:
    """Scout tool - scout a specific location"""
    return f"📍 Scouting {location}... Found enemy patrol."

# Create tool
scout_tool = Tool.create(
    name="Scout",
    func=scout_location,
    description="Scout a location and report findings"
)

# Create agent with tool
scout = Agent(name="Elite Scout", role="Reconnaissance")
scout.add_tool(scout_tool)

# Use the tool
result = scout_tool.use("Enemy camp north")
print(result)
```

### Using Memory

```python
from bingfu import Memory

# Create memory (in-memory mode)
memory = Memory(name="Battle Memory", memory_type="dict")

# Store information
memory.store("enemy_general", "General Wei")
memory.store("our_morale", "High")
memory.store("recommendation", "Quick strike")

# Retrieve information
enemy = memory.retrieve("enemy_general")
print(f"Enemy General: {enemy}")

# Get all data
all_data = memory.all()
for key, value in all_data.items():
    print(f"{key}: {value}")
```

### Sun Tzu Tactics Engine

```python
from bingfu.tactics import TacticsEngine, TacticalContext, SunTzuAgent

# Create tactics engine
engine = TacticsEngine()

# Create battlefield context
context = TacticalContext(
    self_strength=30000,      # Our forces
    enemy_strength=80000,     # Enemy forces
    terrain="Mountain",       # Battlefield terrain
    morale=70                # Morale (0-100)
)

# Analyze battlefield
result = engine.analyze(context)
print(f"Recommended Tactic: {result['recommended_tactic']['strategy']}")
print(f"Tactics:")
for t in result['tactics']:
    print(f"  - {t}")

# Use Sun Tzu Agent
sunbin = SunTzuAgent(name="Sun Tzu", role="Strategist")
advice = sunbin.analyze_battlefield(context)
print(f"Sun Tzu's Advice: {advice['recommended_tactic']['strategy']}")
```

### LangChain Integration

BingFu v0.6.0 introduces LangChain integration for enhanced agent capabilities:

```python
from bingfu import LangChainAgent, LangChainMemory, RAGRetriever

# 1. Create LangChain Agent with multiple memory modes
agent = LangChainAgent(
    name="Zhuge Liang",
    role="Strategist",
    description="Master of strategy and tactics",
    memory_type="summary"  # buffer, summary, window, or vector
)

# 2. Create RAG Retriever for retrieval-augmented generation
retriever = RAGRetriever(
    embedding_model="text-embedding-3-small",
    vector_store_type="faiss"  # faiss or chroma
)

# Add documents to RAG
retriever.add_documents([
    "Sun Tzu's Art of War has 13 chapters.",
    "The supreme art of war is to subdue the enemy without fighting."
])

# 3. Use RAG for knowledge-enhanced queries
results = retriever.similarity_search("What is the best strategy?", k=3)

# 4. Combine Agent with RAG
agent.use_rag = True
agent.rag_retriever = retriever
result = agent.run_with_rag("What does Sun Tzu say about strategy?")
```

### Visualization - Military Command Console

```python
from bingfu.visual import MilitaryCommandConsole

# Create console
console = MilitaryCommandConsole(
    title="BingFu · Military Command",
    width=1200,
    height=800
)

# Add generals
console.add_general("Han Xin", "online", "Commander", "Analyzing battlefield")
console.add_general("Bai Qi", "busy", "General", "Leading vanguard")

# Add reports
console.add_report("Scout Report", "Enemy supply convoy spotted", "success")
console.add_report("Urgent Intel", "Enemy reinforcements incoming", "warning")

# Update battle status
console.update_battle_status(
    own_strength=30000,
    enemy_strength=80000,
    strategy="Outnumbered, use deception"
)

# Run console (blocking)
console.run()
```

## 📚 Examples

All examples are in the `examples/` directory:

| Example | Description |
|---------|-------------|
| `quickstart.py` | 5-minute quickstart guide (7 tutorials) |
| `basic_usage.py` | Basic single-agent usage |
| `multi_agent.py` | Multi-agent coordination & battle scenarios |
| `custom_agent.py` | Custom agents with tools and memory |
| `tool_usage.py` | Complete tool usage guide |
| `memory_usage.py` | Complete memory system guide |
| `famous_generals.py` | Ancient Chinese generals (Han Xin, Bai Qi, etc.) |
| `cli_guide.py` | CLI usage guide |
| `console_demo.py` | Military command console demo |
| `langchain_usage.py` | LangChain integration examples (Agent, Memory, RAG) |

Run examples:

```bash
# Quick start
python examples/quickstart.py

# All basic examples
python examples/basic_usage.py
python examples/multi_agent.py
python examples/custom_agent.py

# Visualization
python examples/console_demo.py
```

## 🏗️ Project Structure

```
bingfu-framework/
├── bingfu/                  # Main package
│   ├── __init__.py          # Package exports
│   ├── agent.py             # Agent class
│   ├── tool.py              # Tool class
│   ├── memory.py            # Memory class
│   ├── signal.py            # Drum/Gong signals
│   ├── commander.py         # Multi-agent coordinator
│   ├── tactics.py           # Sun Tzu tactics engine
│   ├── bingfu.py            # Main BingFu class
│   ├── cli.py               # Command-line interface
│   ├── langchain_integration.py  # LangChain integration
│   ├── llm/                 # LLM providers
│   │   ├── base.py          # Base LLM provider
│   │   ├── openai_provider.py
│   │   └── factory.py       # LLM factory
│   └── visual/              # Visualization module
│       ├── styles.py        # Style constants
│       ├── components.py    # UI components
│       └── console.py       # MilitaryCommandConsole
├── examples/                # Usage examples
├── tests/                   # Unit tests
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

## 📖 API Reference

### Core Classes

#### Agent
```python
agent = Agent(name="Name", role="Role", description="Description")
agent.drum(task)      # Start with task
agent.gong()          # Stop
agent.add_tool(tool)  # Add tool
agent.execute(task)   # Execute task
```

#### Tool
```python
def my_func(x): return x * 2
tool = Tool.create(name="Name", func=my_func, description="Desc")
result = tool.use(5)  # Uses the function
```

#### Memory
```python
memory = Memory(name="Name", memory_type="dict")  # or "file"
memory.store(key, value)      # Store data
value = memory.retrieve(key)  # Retrieve data
data = memory.all()           # Get all data
memory.delete(key)            # Delete data
memory.clear()                # Clear all
```

#### BingFu
```python
bf = BingFu(name="Name")
bf.add_agent(agent)           # Add agent
bf.remove_agent(name)         # Remove agent
bf.drum(name, task)          # Start agent
bf.gong(name)                 # Stop agent
bf.drum_all(task)             # Start all
bf.gong_all()                 # Stop all
bf.enable_commander()         # Enable commander mode
bf.status()                   # Get status
```

#### Commander
```python
commander = Commander(name="Marshal")
commander.add_agent(agent)
commander.coordinate(task)    # Coordinate all
commander.drum_one(name, task)  # Start one
commander.gong_one(name)        # Stop one
```

#### Tactics Engine
```python
engine = TacticsEngine()
context = TacticalContext(
    self_strength=30000,
    enemy_strength=80000,
    terrain="Mountain",
    morale=70
)
result = engine.analyze(context)
```

#### SunTzuAgent
```python
sunbin = SunTzuAgent(name="Sun Tzu", role="Strategist")
result = sunbin.analyze_battlefield(context)
quote = sunbin.get_wisdom()
```

### Visualization

#### MilitaryCommandConsole
```python
console = MilitaryCommandConsole(title="Title", width=1200, height=800)
console.add_general(name, status, role, message)
console.remove_general(name)
console.add_report(title, content, report_type)
console.update_battle_status(own, enemy, strategy)
console.add_tactics(tactic)
console.run()  # or console.run(blocking=False)
console.stop()
```

## 🔗 Links

- **GitHub**: https://github.com/luke99810/bingfu
- **Issues**: https://github.com/luke99810/bingfu/issues

---

# 中文

## 🎯 简介

**兵符 (BingFu)** 是一个受中国古代战争启发的轻量级多智能体框架。它允许你创建**单个智能体**或**多智能体系统**，并使用**战争风格的信号**进行通信：

- **击鼓** — 启动智能体或任务
- **鸣金** — 停止智能体或中止任务

该框架设计简洁、可扩展，并融入了中国军事策略的文化元素。

## ✨ 特性

- **轻量级**：依赖最少，快速上手
- **单/多智能体**：支持独立智能体和协同团队
- **信号系统**：击鼓/鸣金信号，控制流程清晰
- **可扩展**：轻松添加自定义智能体、工具和记忆
- **中国文化特色**：命名和概念源自中国古代战争
- **孙子兵法战术引擎**：内置古代军事智慧
- **古代名将Agent**：白起、韩信、项羽、诸葛亮等预置Agent
- **可视化控制台**：古代军事风格桌面界面（中军帐）

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/luke99810/bingfu.git
cd bingfu

# 安装包（可编辑模式）
pip install -e .
```

## 🚀 快速开始

### 5分钟入门

运行快速入门指南：

```bash
python examples/quickstart.py
```

这将带你学习7个简单示例，掌握所有核心功能。

### 基础用法

```python
from bingfu import BingFu, Agent

# 创建智能体
agent = Agent(name="侦察兵", role="侦察")

# 创建兵符主对象
master = BingFu(name="指挥官")
master.add_agent(agent)

# 发送击鼓信号（开始任务）
result = master.drum("侦察兵", "侦察敌方营地")
print(result)

# 发送鸣金信号（停止）
result = master.gong("侦察兵")
print(result)
```

### 多智能体协作

```python
from bingfu import BingFu, Agent

# 创建兵符
master = BingFu(name="战役指挥部")

# 创建多个智能体
agents = [
    Agent(name="先锋营", role="先锋"),
    Agent(name="主力军", role="主攻"),
    Agent(name="预备队", role="预备"),
]

# 添加所有智能体
for agent in agents:
    master.add_agent(agent)

# 启用指挥官模式
master.enable_commander(name="元帅")

# 击鼓全军（启动所有）
results = master.drum_all("发起进攻")

# 鸣金收兵（停止所有）
results = master.gong_all()
```

### 使用工具

```python
from bingfu import Agent, Tool

# 定义工具函数
def scout_location(location: str) -> str:
    """侦察工具 - 侦察指定位置"""
    return f"📍 正在侦察 {location}... 发现敌军巡逻队。"

# 创建工具
scout_tool = Tool.create(
    name="侦察",
    func=scout_location,
    description="侦察指定地点并返回结果"
)

# 创建带工具的智能体
scout = Agent(name="精锐斥候", role="侦察兵")
scout.add_tool(scout_tool)

# 使用工具
result = scout_tool.use("敌方营地北侧")
print(result)
```

### 使用记忆

```python
from bingfu import Memory

# 创建记忆（内存模式）
memory = Memory(name="战场记忆", memory_type="dict")

# 存储信息
memory.store("敌将", "项羽")
memory.store("我军士气", "高昂")
memory.store("建议", "速战速决")

# 读取信息
enemy = memory.retrieve("敌将")
print(f"敌将: {enemy}")

# 获取所有数据
all_data = memory.all()
for key, value in all_data.items():
    print(f"{key}: {value}")
```

### 孙子兵法战术引擎

```python
from bingfu.tactics import TacticsEngine, TacticalContext, SunTzuAgent

# 创建战术引擎
engine = TacticsEngine()

# 创建战场态势
context = TacticalContext(
    self_strength=30000,      # 我军兵力
    enemy_strength=80000,     # 敌军兵力
    terrain="山地",          # 战场地形
    morale=70                # 士气(0-100)
)

# 分析战场
result = engine.analyze(context)
print(f"推荐战术: {result['recommended_tactic']['strategy']}")
print("具体建议:")
for t in result['tactics']:
    print(f"  - {t}")

# 使用孙子Agent
sunbin = SunTzuAgent(name="孙子", role="军师")
advice = sunbin.analyze_battlefield(context)
print(f"孙子曰: {advice['recommended_tactic']['strategy']}")
```

### 中军帐可视化

```python
from bingfu.visual import MilitaryCommandConsole

# 创建控制台
console = MilitaryCommandConsole(
    title="兵符 · 中军帐",
    width=1200,
    height=800
)

# 添加将领
console.add_general("韩信", "online", "统帅", "正在分析战场形势")
console.add_general("白起", "busy", "主将", "率领先锋突击")

# 添加军情
console.add_report("侦察回报", "发现敌军粮草运输队", "success")
console.add_report("紧急军情", "敌军增援约三万", "warning")

# 更新战役态势
console.update_battle_status(
    own_strength=30000,
    enemy_strength=80000,
    strategy="敌众我寡，宜用奇兵"
)

# 运行控制台（阻塞模式）
console.run()
```

## 📚 示例列表

所有示例都在 `examples/` 目录中：

| 示例文件 | 说明 |
|---------|------|
| `quickstart.py` | 5分钟快速入门（7个教程）|
| `basic_usage.py` | 基础单智能体用法 |
| `multi_agent.py` | 多智能体协作与战役场景 |
| `custom_agent.py` | 自定义智能体（工具+记忆）|
| `tool_usage.py` | 完整工具使用指南 |
| `memory_usage.py` | 完整记忆系统指南 |
| `famous_generals.py` | 古代名将（韩信、白起等）|
| `cli_guide.py` | CLI使用指南 |
| `console_demo.py` | 中军帐可视化演示 |

运行示例：

```bash
# 快速入门
python examples/quickstart.py

# 基础示例
python examples/basic_usage.py
python examples/multi_agent.py
python examples/custom_agent.py

# 可视化
python examples/console_demo.py
```

## 🏗️ 项目结构

```
bingfu-framework/
├── bingfu/                  # 主包
│   ├── __init__.py          # 包导出
│   ├── agent.py             # 智能体类
│   ├── tool.py              # 工具类
│   ├── memory.py            # 记忆类
│   ├── signal.py            # 击鼓/鸣金信号
│   ├── commander.py         # 多智能体协调器
│   ├── tactics.py           # 孙子兵法战术引擎
│   ├── bingfu.py            # 兵符主类
│   ├── cli.py               # 命令行接口
│   └── visual/              # 可视化模块
│       ├── styles.py        # 样式常量
│       ├── components.py   # UI组件
│       └── console.py       # 中军帐控制台
├── examples/                # 使用示例
├── tests/                   # 单元测试
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖
├── CHANGELOG.md            # 版本历史
└── README.md               # 本文档
```

## 🔗 链接

- **GitHub**: https://github.com/luke99810/bingfu
- **Issues**: https://github.com/luke99810/bingfu/issues

---

## 📝 Development Status (开发状态)

| 功能 | 状态 |
|------|------|
| 项目初始化 | ✅ 完成 |
| 核心模块 (Agent, Tool, Memory) | ✅ 完成 |
| 信号系统 (击鼓/鸣金) | ✅ 完成 |
| 指挥官系统 | ✅ 完成 |
| 兵符主类 | ✅ 完成 |
| CLI命令行界面 | ✅ 完成 |
| 孙子兵法战术引擎 | ✅ 完成 |
| 古代名将示例 | ✅ 完成 |
| 扩展示例 | ✅ 完成 |
| 单元测试 | ✅ 完成 |
| 可视化控制台 | ✅ 完成 |
| LangChain集成 (Agent/Memory/RAG) | ✅ 完成 |
| Web版本 | 🔜 未来版本 |

## 📄 License

本项目采用 MIT 许可证 — 详情请参阅 [LICENSE](LICENSE) 文件。
