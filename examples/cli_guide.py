"""
BingFu Framework - CLI使用指南
Command Line Interface Usage Guide

本文件展示 BingFu CLI 的完整使用方法。
使用方式: python -m bingfu.cli <command> [options]

前置条件: pip install -e .
"""

# ==================== 命令概览 ====================
# ==================== Command Overview ====================

# 1. 添加Agent
#    bingfu add-agent <name> [--role <role>]
#
# 2. 发送击鼓信号
#    bingfu drum <agent_name> [--task <task>]
#
# 3. 发送鸣金信号
#    bingfu gong <agent_name>
#
# 4. 查看状态
#    bingfu status
#
# 5. 查看版本
#    bingfu version


# ==================== 详细示例 ====================
# ==================== Detailed Examples ====================

"""
【示例1】添加单个Agent
=========================
$ bingfu add-agent General_Zhang --role "步兵统帅"

输出:
✅ Agent 'General_Zhang' (role: 步兵统帅) added successfully!
"""


"""
【示例2】发送击鼓信号启动Agent
================================
$ bingfu drum General_Zhang --task "率领步兵进攻敌营"

输出:
🥁 Drum signal sent to 'General_Zhang'!
Task: 率领步兵进攻敌营
Agent 'General_Zhang' is now ACTIVE
"""


"""
【示例3】发送鸣金信号停止Agent
==============================
$ bingfu gong General_Zhang

输出:
🔔 Gong signal sent to 'General_Zhang'!
Agent 'General_Zhang' has STOPPED
"""


"""
【示例4】查看所有Agent状态
==========================
$ bingfu status

输出示例:
╔═══════════════════════════════════════╗
║         BingFu Army Status           ║
╠═══════════════════════════════════════╣
║ 🟢  Scouter      - ACTIVE            ║
║ 🟢  Infantry      - ACTIVE            ║
║ 🔴  Cavalry      - STOPPED           ║
║ 🟡  Strategist    - WAITING           ║
╠═══════════════════════════════════════╣
║ Total Agents: 4                      ║
║ Active: 2 | Stopped: 1 | Waiting: 1  ║
╚═══════════════════════════════════════╝
"""


"""
【示例5】击鼓全体
==================
$ bingfu drum-all --task "全军出击"

输出:
🥁 Drum signal sent to ALL agents!
Task: 全军出击
Agents activated:
  - Scouter: ACTIVE
  - Infantry: ACTIVE
  - Cavalry: ACTIVE
  - Strategist: ACTIVE
"""


"""
【示例6】鸣金收兵
=================
$ bingfu gong-all

输出:
🔔 Gong signal sent to ALL agents!
All agents have STOPPED
"""


"""
【示例7】停止单个Agent
======================
$ bingfu gong Infantry

输出:
🔔 Gong signal sent to 'Infantry'!
Agent 'Infantry' has STOPPED
"""


"""
【示例8】查看版本信息
====================
$ bingfu version

输出:
╔═══════════════════════════════════════╗
║           BingFu Framework            ║
║           兵符 · 轻量级多智能体框架       ║
╠═══════════════════════════════════════╣
║ Version: 0.1.0                        ║
║ Author:  SuXin                         ║
║ GitHub:  github.com/luke99810/bingfu  ║
╚═══════════════════════════════════════╝
"""


# ==================== 快捷脚本示例 ====================
# ==================== Quick Script Examples ====================

"""
【脚本1】快速启动战役
=======================
#!/bin/bash
# battle_start.sh

echo "==== 战役开始 ===="

# 添加各路兵马
bingfu add-agent Scout --role "斥候"
bingfu add-agent Infantry --role "步兵"
bingfu add-agent Cavalry --role "骑兵"
bingfu add-agent Archer --role "弓箭手"

# 侦察阶段
bingfu drum Scout --task "侦察敌情"
sleep 2
bingfu gong Scout

# 弓箭压制
bingfu drum Archer --task "弓箭齐射"
sleep 1

# 骑兵突击
bingfu drum Cavalry --task "侧翼突击"
sleep 2

# 步兵推进
bingfu drum Infantry --task "全线推进"
sleep 3

# 鸣金收兵
bingfu gong-all

echo "==== 战役结束 ===="
"""

"""
【脚本2】查看状态监控
=====================
#!/bin/bash
# monitor.sh

while true; do
    clear
    echo "==== 兵符状态监控 ===="
    bingfu status
    echo ""
    echo "按 Ctrl+C 退出"
    sleep 5
done
"""


"""
【脚本3】Python中调用CLI
========================
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu.cli import main
import sys

# 方式1：直接调用main函数
def run_cli_command():
    """在Python中执行CLI命令"""
    # 等同于: bingfu status
    sys.argv = ['bingfu', 'status']
    main()

# 方式2：使用BingFu API
from bingfu import BingFu, Agent

def python_api_example():
    """使用Python API的等效方式"""
    # 创建兵符
    bingfu = BingFu()

    # 添加Agent
    scout = Agent(name="Scout")
    infantry = Agent(name="Infantry")
    bingfu.add_agent(scout)
    bingfu.add_agent(infantry)

    # 查看状态
    print(bingfu.status())

    # 击鼓启动
    bingfu.drum("Scout", "侦察敌情")
    bingfu.drum("Infantry", "进攻")

    # 查看状态
    print(bingfu.status())

    # 鸣金收兵
    bingfu.gong_all()


# ==================== 配置加载示例 ====================
# ==================== Config Loading Example ====================

"""
【配置文件】config.yaml
=======================
core:
  debug: true
  log_level: INFO

agent:
  default_role: "士兵"
  max_retries: 3

signal:
  drum_message: "🥁 击鼓！进军！"
  gong_message: "🔔 鸣金！收兵！"

commander:
  enable_logging: true
  log_file: "battle.log"
"""

"""
【加载配置】
============
$ bingfu load-config config.yaml

输出:
✅ Configuration loaded from 'config.yaml'
✅ Debug mode: ON
✅ Log level: INFO
"""


# ==================== 错误处理 ====================
# ==================== Error Handling ====================

"""
【错误1】Agent不存在
====================
$ bingfu drum NonExistent

输出:
❌ Error: Agent 'NonExistent' not found!
Available agents: Scout, Infantry, Cavalry
"""

"""
【错误2】重复添加Agent
======================
$ bingfu add-agent Scout

输出:
❌ Error: Agent 'Scout' already exists!
Use 'bingfu drum Scout' to activate.
"""


# ==================== 完整工作流 ====================
# ==================== Complete Workflow ====================

"""
【完整战役流程】
================

$ bingfu version
╔═══════════════════════════════════════╗
║           BingFu Framework            ║
║           兵符 v0.1.0                 ║
╚═══════════════════════════════════════╝

# 1. 添加军队
$ bingfu add-agent Commander --role "统帅"
✅ Agent 'Commander' added!

$ bingfu add-agent Scout1 --role "斥候"
✅ Agent 'Scout1' added!

$ bingfu add-agent Scout2 --role "斥候"
✅ Agent 'Scout2' added!

$ bingfu add-agent Infantry --role "步兵"
✅ Agent 'Infantry' added!

$ bingfu add-agent Cavalry --role "骑兵"
✅ Agent 'Cavalry' added!

# 2. 查看状态
$ bingfu status
╔═══════════════════════════════════════╗
║         BingFu Army Status             ║
╠═══════════════════════════════════════╣
║ ⚪ Commander     - STOPPED            ║
║ ⚪ Scout1        - STOPPED           ║
║ ⚪ Scout2        - STOPPED           ║
║ ⚪ Infantry      - STOPPED           ║
║ ⚪ Cavalry       - STOPPED           ║
╠═══════════════════════════════════════╣
║ Total Agents: 5                       ║
║ Active: 0 | Stopped: 5               ║
╚═══════════════════════════════════════╝

# 3. 开始战役
$ bingfu drum Commander --task "制定作战计划"
🥁 Drum signal sent to 'Commander'!
Task: 制定作战计划

$ bingfu drum Scout1 --task "侦察东线"
🥁 Drum signal sent to 'Scout1'!

$ bingfu drum Scout2 --task "侦察西线"
🥁 Drum signal sent to 'Scout2'!

# 4. 收到情报后行动
$ bingfu gong Scout1
🔔 Gong signal sent to 'Scout1'!

$ bingfu gong Scout2
🔔 Gong signal sent to 'Scout2'!

$ bingfu drum Cavalry --task "趁敌军调动突击"
🥁 Drum signal sent to 'Cavalry'!

$ bingfu drum Infantry --task "全线压上"
🥁 Drum signal sent to 'Infantry'!

# 5. 战斗结束
$ bingfu gong-all
🔔 Gong signal sent to ALL agents!

# 6. 最终状态
$ bingfu status
╔═══════════════════════════════════════╗
║         BingFu Army Status             ║
╠═══════════════════════════════════════╣
║ 🔴 Commander     - STOPPED           ║
║ 🔴 Scout1        - STOPPED           ║
║ 🔴 Scout2        - STOPPED           ║
║ 🔴 Infantry      - STOPPED           ║
║ 🔴 Cavalry       - STOPPED           ║
╠═══════════════════════════════════════╣
║ Total Agents: 5                       ║
║ Victory! All enemies defeated!        ║
╚═══════════════════════════════════════╝
"""
