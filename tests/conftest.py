# -*- coding: utf-8 -*-
r"""测试夹具。

════════════════════════════════════════════════════════════════
 为什么控制台必须全会话只有一个
════════════════════════════════════════════════════════════════

同一个进程里建多个 Tk 根会互相破坏状态，症状是一句
**完全指不到真因**的报错：

    _tkinter.TclError: Can't find a usable tk.tcl ...
    This probably means that tk wasn't installed properly.

而且它是**顺序相关**的：先跑 GUI 测试 A 再跑 GUI 测试 B 会炸，
反过来却不会。于是单独跑某个文件永远正常，全套跑才出问题 ——
这类飘忽的失败最危险，因为人会学会「重跑一次就绿了」，
然后把真实的失败也一起无视掉。

``test_visual.py`` 早就发现了这件事，把它的 console 夹具做成了
session 级。但夹具定义在那个文件里，只对它自己可见 ——
后来新增的 ``test_i18n.py`` 需要窗口，自己又建了一个，
两边就撞上了。

★ 夹具放进 conftest.py，是让「全会话唯一」这句话对**所有**测试成立，
  而不只是对当初写它的那个文件成立。
"""

import pytest


@pytest.fixture(scope="session")
def console():
    """全会话唯一的控制台实例。

    ★ 谁都不要在自己的测试里另起一个 ``MilitaryCommandConsole``。
      需要窗口就取这个夹具；需要干净状态就用 ``reset_console`` 复位。
    """

    tk = pytest.importorskip("tkinter")
    try:
        from bingfu.visual import MilitaryCommandConsole
        instance = MilitaryCommandConsole(title="测试控制台",
                                          width=1200, height=800)
    except tk.TclError as exc:          # 无显示环境
        pytest.skip("没有可用的显示环境：%s" % exc)
    yield instance
    try:
        instance.stop()
    except Exception:                   # noqa: BLE001
        pass


@pytest.fixture
def reset_console(console):
    """把共用窗口复位到「刚建好」的状态，用完再还原。

    ★ 复用窗口的代价是测试之间会互相污染，所以复位必须彻底 ——
      上一条测试加的将领会让下一条的断言莫名其妙地通过或失败。

    ★ 还原 ``title`` 是必须的：本夹具会把它改成当前语言的应用名，
      而 ``test_visual`` 断言的是构造时传入的「测试控制台」。
      不还原就等于一个测试悄悄改坏了另一个测试。
    """

    from bingfu import i18n

    original_title = console.title

    def _prepare():
        i18n.reset_for_tests()
        console.title = i18n.t("app.title")
        for name in list(getattr(console, "generals", {}) or {}):
            try:
                console.remove_general(name)
            except Exception:           # noqa: BLE001
                console.generals.pop(name, None)
        if hasattr(console, "_refresh_generals"):
            console._refresh_generals()
        console.update_battle_status(0, 0, None)
        console.retranslate()
        console._root.update()
        return console

    yield _prepare
    i18n.reset_for_tests()
    console.title = original_title
    try:
        console.retranslate()
        console._root.update()
    except Exception:                   # noqa: BLE001
        pass
