"""
兵符 · 中军帐 启动入口
一键启动 LLM 驱动的可视化控制台 + 智能派兵系统

★ 版本号不写在这里 —— 以 bingfu.__version__ 为准。
  此前全项目有四个不同的版本号（pyproject 0.6.0 / __init__ 0.6.0 /
  BingFu 类 0.5.0 / config.yaml 0.1.0），而这个文件到处打印 v0.5.0。

兵法云：将听吾计，用之必胜——军师就位，方能运筹帷幄。
知人善任，点兵有据——选对将领，战事已半。
"""

import io
import os
import sys

# ★ 先把标准输出切成 UTF-8，再做任何 print。
#
#   中文 Windows 的控制台默认是 GBK，而本文件里有 emoji（🧠 🎖️ 📯 🥁）。
#   GBK 编不了它们，print 直接抛 UnicodeEncodeError ——
#   程序在「军师初始化中…」那一行崩掉，而崩溃信息看起来像是
#   LLM 初始化出了问题，跟真正的原因（控制台编码）毫无关系。
#
#   输出被重定向到文件时同样会触发：那种情况下 Python 用的是
#   locale 编码，仍然是 GBK。
#
#   errors="replace" 是有意的：编不出来的字符显示成问号即可，
#   不值得为一个装饰性的 emoji 让整个程序起不来。
for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ★ 确保可以导入 bingfu。
#
#   原先写的是 dirname(dirname(...)) —— 指向 BingFuAgent/，**再上一层**，
#   那底下根本没有 bingfu 包。它一直没出事，只是因为直接运行脚本时
#   Python 会自动把脚本所在目录加进 sys.path，
#   也就是说那行 insert 从来没起过作用（而看起来像在起作用）。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bingfu import BingFu, Agent, __version__
from bingfu.llm import LLMFactory, LLMConfig
from bingfu.profile import GeneralProfile, CombatStats, CombatStyle
from bingfu.presets import get_preset
from bingfu.visual.console import MilitaryCommandConsole
from bingfu.tools import belt_for, workspace_at
from bingfu.memory import EpisodicMemory, KnowledgeBase
from bingfu.checkpoint import JSONCheckpointer
from bingfu.i18n import t as tr

# ★ 命令执行默认关闭。
#
#   run_command 的能力边界就是当前用户的权限边界，没有沙箱。
#   要开就显式开：启动前设 BINGFU_ENABLE_SHELL=1。
#   默认值选「关」，是因为这个选择的两种错法代价完全不对等 ——
#   少一个工具只是不方便，多一个无边界的工具是另一回事。
ENABLE_SHELL = os.environ.get("BINGFU_ENABLE_SHELL", "").strip() in ("1", "true", "yes")


#: 凭据的查找顺序。
#:
#: ★ 顺序固定且**对外可见**，是因为「有好几个地方都可能放着 key」
#:   这件事本身就容易出错：使用者改了其中一处，程序却在用另一处，
#:   而两者看起来都「已配置」。
#:
#:   环境变量排第一：CI 与临时切换靠它，且它比落盘的明文更容易撤销。
KEY_SOURCES = [
    ("环境变量 DEEPSEEK_API_KEY", "env:DEEPSEEK_API_KEY"),
    ("项目 .env 的 DEEPSEEK_API_KEY", "dotenv:DEEPSEEK_API_KEY"),
    ("~/.openharness/config.toml 的 api_key", "toml"),
    ("环境变量 OPENAI_API_KEY（作为兜底）", "env:OPENAI_API_KEY"),
    ("项目 .env 的 OPENAI_API_KEY（作为兜底）", "dotenv:OPENAI_API_KEY"),
]


def _dotenv_values():
    """读项目目录下的 .env。

    ★ 用 dotenv_values 而不是 load_dotenv()。

      load_dotenv 会把值灌进 os.environ —— 那会让「这个 key 从哪来的」
      变得不可追溯，也会污染子进程（工具层要起 subprocess）。
      这里只是读出来看看，由上面那张表决定谁优先。

    ★ python-dotenv 没装时不报错，返回空 —— 它只是几个来源之一。
    """

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return {}
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v}
    except ImportError:
        # 手工解析：装不装 python-dotenv 都不该影响能不能读到 key
        out = {}
        try:
            for line in io.open(path, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    out[k.strip()] = v
        except OSError:
            pass
        return out
    except Exception:
        return {}


def _toml_api_key():
    path = os.path.expanduser("~/.openharness/config.toml")
    if not os.path.exists(path):
        return "", ""
    try:
        try:
            import tomllib          # Python 3.11+ 标准库
        except ImportError:
            import tomli as tomllib  # 老版本靠 tomli
    except ImportError:
        # ★ 原来这里 `except ImportError: tomllib = None` 之后整段跳过 ——
        #   在 3.11+ 上标准库明明有 tomllib，却因为只试了 tomli 而放弃。
        return "", ""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return str(data.get("api_key", "") or ""), str(data.get("model", "") or "")
    except Exception:
        return "", ""


def resolve_api_key():
    """按固定顺序找 key。返回 (key, 来源说明, 模型名)。"""

    dot = _dotenv_values()
    toml_key, toml_model = _toml_api_key()

    for label, spec in KEY_SOURCES:
        if spec.startswith("env:"):
            v = os.environ.get(spec[4:], "").strip()
        elif spec.startswith("dotenv:"):
            v = str(dot.get(spec[7:], "") or "").strip()
        elif spec == "toml":
            v = toml_key.strip()
        else:
            v = ""
        if v:
            model = toml_model if (spec == "toml" and toml_model) else "deepseek-chat"
            return v, label, model
    return "", "", "deepseek-chat"


def describe_key_sources():
    """每一处「有没有」—— 只说有无，不显示值。"""

    dot = _dotenv_values()
    toml_key, _ = _toml_api_key()
    lines = []
    for label, spec in KEY_SOURCES:
        if spec.startswith("env:"):
            has = bool(os.environ.get(spec[4:], "").strip())
        elif spec.startswith("dotenv:"):
            has = bool(str(dot.get(spec[7:], "") or "").strip())
        elif spec == "toml":
            has = bool(toml_key.strip())
        else:
            has = False
        lines.append(f"{'✓' if has else '×'} {label}")
    return lines


def main():
    print("=" * 60)
    print(f"  兵符 · 中军帐  Multi-Agent 战役指挥系统 v{__version__}")
    print("  新增功能：将军战力体系 + 智能派兵")
    print("=" * 60)
    print()

    # ===== 1. 创建 BingFu 框架 =====
    master = BingFu(name="兵符")   # 版本由包统一提供

    # ===== 2. 初始化 LLM（军师） =====
    api_key, key_source, model = resolve_api_key()

    if api_key:
        print(f"🧠 军师初始化中...（凭据来自：{key_source}）")
        llm_config = LLMConfig(
            provider="deepseek",
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=2048,
        )
        llm = LLMFactory.create(llm_config)
        master.set_llm(llm)
        print(f"🧠 军师就位：DeepSeek ({model})")
    else:
        # ★ 说清楚「去哪几个地方找过」，而不是只说没找到。
        #
        #   原来只报「未检测到 DEEPSEEK_API_KEY」，而实际上它还会去读
        #   ~/.openharness/config.toml —— 使用者按提示去设环境变量，
        #   设完发现另一处的旧 key 仍在生效，或者反过来：
        #   .env 里明明有 key，程序却说没有（因为它根本不读 .env）。
        print("⚠️  没有找到可用的 API Key，将领将以关键词匹配模式运作。")
        print("    找过这几处（按优先级）：")
        for line in describe_key_sources():
            print(f"      {line}")
        llm = None

    # ===== 3. 创建将领 Agent（带预设战力档案）=====
    # 从预设库加载档案
    # ★ 初始一律「待命」。
    #
    #   原来这里写的是 online / busy / online / idle —— 白起「作战中」、
    #   项羽「空闲」，而此时一道军令都还没下，谁也没在做任何事。
    #   一个凭空写死的状态字段，会让人以为它反映的是真实情况；
    #   等到真有将领在执行时，反而分不清哪个是真的在忙。
    #
    #   现在这个字段由战役流程驱动：受命时转「作战中」，复命后转回「待命」。
    generals_setup = [
        ("韩信", "统帅", "汉初三杰之一，善于统兵作战，出奇制胜", "idle"),
        ("白起", "主将", "战国四大名将之首，善于歼灭战", "idle"),
        ("诸葛亮", "军师", "卧龙先生，善于谋略推演", "idle"),
        ("项羽", "虎将", "西楚霸王，勇冠三军", "idle"),
    ]

    # ★ 兵器谱：将领必须拿到工具，否则这个框架只会说话。
    #
    #   此前这里创建 Agent 时不传任何 tools，`_tool_functions` 为空，
    #   `_build_tool_definitions()` 返回空列表 —— ReAct 循环里
    #   `has_tool_calls` 恒为假，**只跑一轮就返回**。
    #
    #   外面看到的仍是一段像模像样的军令式回复，所以这个失效极难察觉：
    #   它不报错、不为空、语气也对，只是从未触碰过这台电脑。
    #
    #   工作区默认设在项目根目录（launch.py 的上一层），
    #   所有文件操作都被限制在其中 —— 一个把相对路径算错的将领
    #   不应该有能力覆盖掉框架自己的源码。
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ws = workspace_at(workspace_root, enable_shell=ENABLE_SHELL)

    # 记忆：战报库与知识库落在工作区下的 .bingfu/
    #
    # ★ 这两行以前不存在，于是 Agent.episodic / .knowledge 恒为 None。
    #
    #   代码是齐的：agent.py 里回放与记录的分支都写好了，
    #   memory/ 下三层共 545 行也都实现了 —— 唯独没人把库建出来传进去。
    #   表现是「跑完什么都不留、下一仗从零开始」，而全程不报任何错。
    #   一个到不了的能力，与不存在的区别只在文档里。
    #
    # ★ 放在工作区下而不是用户目录：换个工作区就是换一段记忆，
    #   两个不相干的项目不该共用一份历史。
    mem_dir = os.path.join(ws.root, ".bingfu")
    os.makedirs(mem_dir, exist_ok=True)
    episodic = EpisodicMemory(path=os.path.join(mem_dir, "episodes.json"))
    # 断点：一次跑了十分钟的战役崩在最后一个节点上，不该整个作废
    checkpointer = JSONCheckpointer(os.path.join(mem_dir, "checkpoints.json"))
    knowledge = KnowledgeBase(path=os.path.join(mem_dir, "knowledge.json"))

    belt = belt_for("Data", knowledge=knowledge)   # Data 类同时开 web 与 code

    tool_funcs = dict(belt.as_functions())
    tool_funcs.update(ws.as_functions())
    tool_descs = dict(ws.descriptions())

    print(f"  🗡️  兵器谱：{'、'.join(tool_funcs)}")
    print(f"  🏰 工作区：{ws.root}"
          f"{'（已开启命令执行）' if ENABLE_SHELL else '（命令执行未开启）'}")

    for name, role, desc, _ in generals_setup:
        try:
            # 尝试从预设库获取档案
            profile = get_preset(name)
        except KeyError:
            # 无预设则使用通用档案
            profile = GeneralProfile(
                style=CombatStyle.COMMAND,
                specialties=["综合分析"],
                weaknesses=[],
                stats=CombatStats(),
                bio=desc,
            )

        agent = Agent(
            name=name,
            role=role,
            description=desc,
            profile=profile,
            llm=llm,
            episodic=episodic,
            knowledge=knowledge,
            system_prompt=(
                f"你是{name}，{desc}。"
                f"作战风格：{profile.style.value}。"
                f"专长：{'、'.join(profile.specialties)}。"
                f"弱项：{'、'.join(profile.weaknesses)}。"
                f"\n\n你有真实的工具可以操作这台电脑（列目录、读写文件、查找文件、执行代码）。"
                f"需要了解或改动实际文件时，**直接调用工具去做**，不要只描述计划、"
                f"也不要凭想象编造文件内容 —— 编出来的内容看起来和真的一样，"
                f"而使用者无从分辨。"
                # ★ 界面语言必须传导到模型。
                #
                #   只切界面的话，英文界面里会蹦出整段中文军令式回复 ——
                #   比不做还难看。这一行让「切换语言」成为一件完整的事。
                f"\n{tr('prompt.reply_language')}"
            ),
        )

        # 装配兵器
        #
        # ★ 分两类注册，不是图省事：
        #
        #   工作区工具（列目录 / 读写文件 / 查找）注册为**随身兵器** ——
        #   编排层会按子任务类别清空重配工具，随身兵器每次都会被重新装上。
        #   文件读写不是「某一类任务的附加能力」：写作任务同样要读素材、
        #   要把稿子落盘。
        #
        #   检索与代码执行则按类别配发，交给路由表决定 ——
        #   给写作任务塞一个 Python 解释器只会增加跑偏的机会，
        #   而且每一轮都要把工具定义塞进上下文。
        for tname, tfunc in ws.as_functions().items():
            agent.register_base_tool(
                tname, tfunc, description=tool_descs.get(tname, tname))
        for tname, tfunc in belt.as_functions().items():
            agent.register_tool_function(tname, tfunc)

        master.add_agent(agent)
        print(f"  🎖️  {name}（{role}）已就位 — {profile}"
              f"｜兵器 {len(tool_funcs)} 件")

    # 启用指挥系统（解锁智能派兵）
    master.enable_commander(name="中军帐")
    print(f"  📯 指挥系统已启用：{master.commander}")

    # ===== 4. 创建可视化控制台 =====
    console = MilitaryCommandConsole(
        title="兵符 · 中军帐",   # ★ 版本号不进标题：它是运行信息，不是身份
        width=1200,
        height=800,
        llm_provider=llm,
        bingfu_instance=master,
        checkpointer=checkpointer,
    )

    # 同步 Agent 状态到控制台 UI
    status_map = {
        "online": "online",
        "busy": "busy",
        "idle": "idle",
        "offline": "offline",
    }
    for name, role, desc, status_text in generals_setup:
        console.add_general(name, status_map.get(status_text, "online"), role, desc)

    # ★ 只报**真实的系统状态**，不编军情。
    #
    #   原来这里预置了三条：「敌军粮草队已过乌江」「发现敌军增援约三万人」
    #   「我军粮草尚可支撑七日」—— 全是虚构的。它们与战役流程真正产出的
    #   军情长得一模一样，混在同一个面板里，使用者无从分辨哪条是系统
    #   算出来的、哪条是摆设。
    console.add_report(
        tr("boot.ready"),
        f"兵符 v{__version__}｜{len(generals_setup)} 位将领待命"
        f"｜兵器 {len(tool_funcs)} 件｜工作区 {ws.root.name}"
        f"｜战报 {len(episodic)} 条｜知识 {len(knowledge)} 条",
        "success")
    console.add_report(
        tr("boot.awaiting"), tr("boot.awaiting_detail"), "info")

    # 战役态势：未受命时敌我皆为 0
    #
    # ★ 0 不是「没数据」的占位，它就是此刻的真值：还没有任务，
    #   没有敌方难度可估，也没有将领出兵。
    console.update_battle_status(0, 0, None)   # None = 沿用默认文案，跟随语言

    # 战术建议留空 —— 它应当由战役产出，而不是开机就摆四句兵法。
    console.clear_tactics()

    # ===== 5. 启动！ =====
    print()
    print("🥁 中军帐已升起，输入指令开始指挥！")
    print()
    print("  【新功能：将军战力档案 + 智能派兵】")
    print("  /match <任务>  — 点兵，展示所有将领匹配评分")
    print("  /smart <任务>  — 智能派兵，自动选最优将领执行")
    print()
    print("  自然语言示例：")
    print("    让系统自动选择合适的将领去完成任务")
    print("    /smart 分析这份数据报告并给出建议")
    print()
    print("=" * 60)

    console.run()


if __name__ == "__main__":
    main()
