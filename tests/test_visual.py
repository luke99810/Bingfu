"""
Visual Module Tests (可视化模块测试)
Tests for the MilitaryCommandConsole visualization module.
Note: These tests skip actual window creation (Tkinter).
"""

import pytest
from bingfu.visual.styles import (
    COLORS, FONTS, ICONS, BORDER_DOUBLE,
    CHINESE_NUMBERS, to_chinese_number
)


# ══════════════════════════════════════════════════════════════
#  ★ 整个测试会话**只允许一个 Tk 根**
# ══════════════════════════════════════════════════════════════
#
#  Tkinter 的真实约束是「一个进程一个根窗口」。原先有 6 条测试各建各的，
#  于是失败位置随运行顺序在三条不同的测试之间漂移 ——
#  单独跑 test_visual.py 挂 A，跟全套一起跑挂 B，重跑一次又全绿。
#
#  ★ 那种测试比稳定失败的更危险：人会学会「重跑一次就行」，
#    然后把真实的失败也一起无视掉。
#
#  修法不是加重试，是让这个约束**在结构上无法被违反**：
#  全会话共用一个 console，谁都不再自己 new。
#
#  （产品侧的真因已单独修掉：console.stop() 原先只 quit() 不 destroy()，
#    Tk 根一直泄漏 —— 那是「开控制台→关掉→再开」必然出错的产品缺陷。）


# ★ console 夹具已挪到 tests/conftest.py。
#
#   放在这里时它只对本文件可见，于是后来新增的 test_i18n.py
#   自己又建了一个 Tk 根 —— 两个根撞在一起，报一句
#   「tk wasn't installed properly」，而且是顺序相关的。
#   放进 conftest 才让「全会话唯一」对所有测试都成立。


class TestStyles:
    """Test styles module"""

    def test_colors_exist(self):
        """Test all required colors are defined"""
        required_colors = [
            "bg_dark", "bg_medium", "bg_light",
            "gold", "gold_bright", "gold_dark",
            "text_primary", "text_secondary",
            "status_online", "status_busy", "status_offline"
        ]
        for color in required_colors:
            assert color in COLORS
            assert COLORS[color].startswith("#")

    def test_fonts_exist(self):
        """Test all required fonts are defined"""
        required_fonts = ["title", "subtitle", "body", "mono", "small"]
        for font in required_fonts:
            assert font in FONTS

    def test_icons_exist(self):
        """Test all required icons are defined"""
        required_icons = ["general", "battle", "report", "drum", "bell"]
        for icon in required_icons:
            assert icon in ICONS

    def test_chinese_numbers(self):
        """Test Chinese number conversion"""
        assert to_chinese_number(0) == "零"
        assert to_chinese_number(1) == "一"
        assert to_chinese_number(5) == "五"
        assert to_chinese_number(10) == "十"


class TestConsoleAPI:
    """Test MilitaryCommandConsole API without creating windows"""

    def test_console_import(self):
        """Test console can be imported"""
        from bingfu.visual import MilitaryCommandConsole
        assert MilitaryCommandConsole is not None

    def test_console_creation_api(self, console):
        """构造参数必须被如实保存下来。"""
        assert console.title == "测试控制台"
        assert console.width == 1200
        assert console.height == 800

    # ══════════════════════════════════════════════════════════
    #  ★ 方法存在性：查**类**，不开窗口
    # ══════════════════════════════════════════════════════════
    #
    #  这四条原先每条都真的 `MilitaryCommandConsole(...)` 开一个 Tk 窗口，
    #  只为断言某个方法名存在。后果是一个**飘的测试**：
    #
    #    单独跑 test_visual.py  → test_add_general_api 失败
    #    跟全套一起跑           → test_update_battle_status_api 失败
    #
    #  同一个进程里建多个 Tk() 根窗口会互相破坏状态，于是报出
    #  「This probably means that tk wasn't installed properly」——
    #  一句完全指不到真因的话。
    #
    #  ★ 失败位置会漂移的测试比稳定失败的更糟：人会学会「重跑一次就绿了」，
    #    然后连真实的失败也一起无视掉。
    #
    #  而这四条要证明的事（方法存在）**根本不需要窗口**：问类就够了。

    @pytest.mark.parametrize(
        "method",
        ["add_general", "add_report", "update_battle_status", "add_tactics"],
    )
    def test_public_api_exists_on_the_class(self, method):
        """不构造实例 —— 既不飘，也不需要图形环境。"""
        from bingfu.visual import MilitaryCommandConsole

        assert callable(getattr(MilitaryCommandConsole, method, None)), (
            f"MilitaryCommandConsole 没有 {method}() 方法"
        )


class TestComponents:
    """Test UI component classes"""

    def test_general_card_import(self):
        """Test GeneralCard can be imported"""
        from bingfu.visual.components import GeneralCard
        assert GeneralCard is not None

    def test_battle_status_panel_import(self):
        """Test BattleStatusPanel can be imported"""
        from bingfu.visual.components import BattleStatusPanel
        assert BattleStatusPanel is not None

    def test_report_panel_import(self):
        """Test ReportPanel can be imported"""
        from bingfu.visual.components import ReportPanel
        assert ReportPanel is not None

    def test_stats_bar_import(self):
        """Test StatsBar can be imported"""
        from bingfu.visual.components import StatsBar
        assert StatsBar is not None

    def test_command_input_import(self):
        """Test CommandInput can be imported"""
        from bingfu.visual.components import CommandInput
        assert CommandInput is not None

    def test_styled_frame_import(self):
        """Test StyledFrame can be imported"""
        from bingfu.visual.components import StyledFrame
        assert StyledFrame is not None


class TestQuickFunctions:
    """Test quick utility functions"""

    def test_create_console_function(self):
        """★ 只验证工厂函数确实产出 MilitaryCommandConsole，**不再另开一个 Tk 根**。

        原先这条自己 new 了第二个控制台，是飘的主要来源之一。
        「工厂返回的是不是那个类」用签名与实现就能判定，不需要真开窗口。
        """
        import inspect

        from bingfu.visual import MilitaryCommandConsole
        from bingfu.visual.console import create_console

        assert callable(create_console)
        source = inspect.getsource(create_console)
        assert MilitaryCommandConsole.__name__ in source, (
            "create_console 不再产出 MilitaryCommandConsole —— 契约已变"
        )

    def test_launch_demo_function_exists(self):
        """Test launch_demo function exists"""
        from bingfu.visual.console import launch_demo
        assert callable(launch_demo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
