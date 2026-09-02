# -*- coding: utf-8 -*-
r"""中英切换的测试。

════════════════════════════════════════════════════════════════
 最容易写成空检查的一条
════════════════════════════════════════════════════════════════

「切换后界面还在、没报错」是空检查 —— 什么都不做也能通过。
必须断言的是：**同一个 widget 上的文字真的变了**，
而且变成了目标语言。所以下面会把 widget 的 text 读回来比对。

════════════════════════════════════════════════════════════════
 窗口从哪来
════════════════════════════════════════════════════════════════

★ 不在本文件里建窗口。

  窗口由 tests/conftest.py 的 session 级 ``console`` 夹具提供，
  全测试会话只有一个；``reset_console`` 负责复位。

  自己再建一个 Tk 根的话，两个根会互相破坏状态，
  而报错会出现在**别的文件**里（test_visual 的建窗测试），
  报一句「tk wasn't installed properly」，根本指不到这里。
  这个坑本文件踩过：症状是顺序相关、单独跑必绿的飘忽失败。
"""

import glob
import io
import os

import pytest

from bingfu import i18n
from bingfu.i18n import CATALOG, LOCALES, get_locale, missing_keys, set_locale, t


@pytest.fixture(autouse=True)
def _clean():
    i18n.reset_for_tests()
    yield
    i18n.reset_for_tests()


# ════════════════════════════════════════════════════════════════
#  一、词条本身
# ════════════════════════════════════════════════════════════════

def test_every_key_has_both_languages():
    """★ 漏翻要在测试里被抓住，而不是等英文界面里蹦出中文。"""

    missing = {k: [loc for loc in LOCALES if not v.get(loc)]
               for k, v in CATALOG.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, "以下词条缺语言：%r" % missing


def test_no_english_entry_is_just_the_chinese_copied():
    """英文词条不能是中文原样复制 —— 那是「翻了」的假象。"""

    def has_cjk(x):
        return any("一" <= ch <= "鿿" for ch in x)

    bad = [k for k, v in CATALOG.items() if has_cjk(v["en"])]
    # app.lang_button 的 en 值是「中」，那是按钮上的字形，属于例外
    bad = [k for k in bad if k != "app.lang_button"]
    assert not bad, "这些词条的英文里还有中文：%r" % bad


def test_unknown_locale_raises_instead_of_being_ignored():
    """★ 不认识的语言代码不能静默忽略。

    静默保持原样的话，调用方以为切过去了，
    而「界面没变」的原因永远查不出来。
    """

    with pytest.raises(ValueError):
        set_locale("fr")
    assert get_locale() == "zh"


def test_missing_key_is_recorded_not_silently_swallowed():
    assert t("nope.not.a.key") == "nope.not.a.key"
    assert "nope.not.a.key" in missing_keys()


def test_switching_changes_what_t_returns():
    assert t("panel.generals") == "将领名录"
    set_locale("en")
    assert t("panel.generals") == "Roster"


def test_observers_are_notified():
    seen = []
    i18n.on_change(lambda loc: seen.append(loc))
    set_locale("en")
    assert seen == ["en"]


def test_a_broken_observer_does_not_abort_the_switch():
    """★ 某个组件重绘失败，不该让切换半途而废 ——
    那会留下一个一半中文一半英文的界面。"""

    order = []
    i18n.on_change(lambda loc: order.append("first"))
    i18n.on_change(lambda loc: (_ for _ in ()).throw(RuntimeError("boom")))
    i18n.on_change(lambda loc: order.append("third"))
    set_locale("en")
    assert order == ["first", "third"], "坏掉的观察者挡住了后面的"
    assert get_locale() == "en"


# ════════════════════════════════════════════════════════════════
#  二、界面真的跟着变了
# ════════════════════════════════════════════════════════════════

def test_toggle_actually_retranslates_the_widgets(reset_console):
    """★ 整组里最要紧的一条：读回 widget 的文字比对。"""

    c = reset_console()
    assert "将领名录" in c.generals_title.cget("text")
    assert c.battle_panel.own_title.cget("text") == "己方兵力"
    assert c.command_input.submit_button.cget("text") == "传达"

    c.toggle_language()
    c._root.update()

    assert get_locale() == "en"
    assert "Roster" in c.generals_title.cget("text"), \
        "面板标题没跟着切：%r" % c.generals_title.cget("text")
    assert c.battle_panel.own_title.cget("text") == "Our Strength"
    assert c.command_input.submit_button.cget("text") == "Send"
    assert c.report_panel.clear_button.cget("text") == "Clear"
    assert c.tactics_frame.title_label.cget("text").strip() \
        == "Art of War · Tactical Advice"
    assert c._root.title() == "BingFu · Command Tent"

    c.toggle_language()
    c._root.update()
    assert "将领名录" in c.generals_title.cget("text"), "切不回来"


def test_the_visible_main_title_switches_too(reset_console):
    """★ 这条是实机截图补出来的。

    原来只断言了 ``root.title()``（窗口标题栏），而界面正中那个
    大字标题是另一个 widget —— 它当时是个局部变量，切换语言时
    根本改不到。测试全绿，截图上却明明白白挂着「兵符 · 中军帐」。

    **只断言到得了的地方，等于只测了自己记得的那部分。**
    """

    c = reset_console()
    assert c.title_label.cget("text") == "兵符 · 中军帐"
    c.toggle_language()
    c._root.update()
    assert c.title_label.cget("text") == "BingFu · Command Tent", \
        "界面正中的主标题没跟着切：%r" % c.title_label.cget("text")


def test_general_status_words_switch_but_names_do_not(reset_console):
    """将领的状态词是界面词汇，名号与简介是数据。"""

    c = reset_console()
    c.add_general("韩信", "idle", "统帅", "汉初三杰之一")
    c._root.update()
    c.toggle_language()
    c._root.update()

    texts = []

    def walk(w):
        for ch in w.winfo_children():
            try:
                txt = ch.cget("text")
            except Exception:           # noqa: BLE001
                txt = ""
            if txt:
                texts.append(str(txt))
            walk(ch)

    walk(c.generals_frame)
    joined = " | ".join(texts)
    assert "Standby" in joined, "状态词没切成英文：%r" % joined
    assert "待命" not in joined, "还留着中文状态词：%r" % joined
    assert "韩信" in joined, "将领名号被翻译了 —— 那是数据，不该动"


def test_lang_button_shows_the_destination_not_the_current(reset_console):
    """按钮是动作，写当前语言会让人按反。"""

    c = reset_console()
    assert c.lang_button.cget("text") == "EN"   # 中文界面 → 按了变英文
    c.toggle_language()
    c._root.update()
    assert c.lang_button.cget("text") == "中"


def test_computed_strategy_is_not_overwritten_by_a_language_switch(reset_console):
    """★ 战役算出来的结论不能被切语言换成默认文案。

    默认文案该跟着语言走，模型/战役的产出不该被翻译 ——
    否则切一次语言就把一句真实结论抹成了「尚未受领军令」，
    而界面上看起来毫无异样。
    """

    c = reset_console()
    c.update_battle_status(120, 70, "敌众我寡，宜用奇兵")
    c._root.update()
    c.toggle_language()
    c._root.update()
    assert c.battle_panel.strategy_label.cget("text") == "敌众我寡，宜用奇兵"
    assert c.battle_panel.strategy_title.cget("text") == "Strategy"


def test_default_strategy_passed_as_none_keeps_following_the_language(reset_console):
    """★ 调用方传 None 表示「仍是默认」，传字符串才算战役产出。

    launch.py 原本写死一句中文传进来，于是默认文案被当成结论冻住 ——
    英文界面的战略栏一直挂着中文。
    """

    c = reset_console()
    c.update_battle_status(0, 0, None)
    c.toggle_language()
    c._root.update()
    assert c.battle_panel.strategy_label.cget("text") == "No order received yet"


def test_lang_command_accepts_explicit_locale(reset_console):
    c = reset_console()
    c._cmd_lang("en")
    assert get_locale() == "en"
    c._cmd_lang("zh")
    assert get_locale() == "zh"
    c._cmd_lang("de")          # 未知代码：不切，也不崩
    assert get_locale() == "zh"


# ════════════════════════════════════════════════════════════════
#  三、后端也跟着切
# ════════════════════════════════════════════════════════════════

def test_campaign_event_titles_follow_the_locale():
    """只切界面不切事件标题，等于英文界面里夹着中文阶段名。"""

    from bingfu import campaign as camp_mod

    assert camp_mod.tr("campaign.assess").startswith("庙算")
    set_locale("en")
    assert "Assessment" in camp_mod.tr("campaign.assess")


def test_prompt_language_instruction_switches():
    """★ 模型回复的语言由系统提示决定 —— 这条不切，就只是「界面翻译」。"""

    assert "中文" in t("prompt.reply_language")
    set_locale("en")
    assert "English" in t("prompt.reply_language")


def _launch_source():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return io.open(os.path.join(here, "launch.py"), encoding="utf-8").read()


def test_launch_passes_the_language_instruction_into_the_prompt():
    """结构性检查：拦得住「有人把这一行删了」。"""

    assert "prompt.reply_language" in _launch_source(), \
        "系统提示里没有语言指示，界面切了模型也不会跟着切"


def test_launch_does_not_hardcode_the_default_strategy():
    assert 'update_battle_status(0, 0, "尚未受领军令")' not in _launch_source(), \
        "launch.py 又把默认战略写死成中文了"


# ════════════════════════════════════════════════════════════════
#  四、守住「全会话只有一个窗口」这条规矩
# ════════════════════════════════════════════════════════════════

def test_no_test_file_creates_its_own_console():
    """★ 把这条规矩本身变成会失败的检查。

    任何测试文件里出现 ``MilitaryCommandConsole(`` 都会再建一个 Tk 根，
    而后果会落在**别的文件**上、报一句指不到真因的话。
    唯一的例外是 conftest.py 里那个 session 级夹具。
    """

    # ★ 用 AST 找**真正的调用**，不要字符串匹配。
    #
    #   第一版是 `"MilitaryCommandConsole(" in src`，结果它把自己也算了
    #   进去 —— 本函数的文档字符串和代码里都出现了这个字面量。
    #   一个把自己判成违规的检查，不是严格，是不准。
    import ast

    here = os.path.dirname(os.path.abspath(__file__))
    offenders = []
    for path in glob.glob(os.path.join(here, "test_*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name == "MilitaryCommandConsole":
                    offenders.append("%s:%d" % (os.path.basename(path),
                                                node.lineno))
    assert not offenders,         "这些测试自己建了控制台，请改用 console/reset_console 夹具：%r" % offenders
