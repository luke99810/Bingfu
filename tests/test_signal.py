"""
Tests for Signal module (Signal 模块测试)
"""

import pytest
from bingfu.signal import Signal, Drum, Gong, SignalType, drum, gong


class TestSignal:
    """Test cases for Signal classes (Signal 类测试用例)."""
    
    def test_create_drum_signal(self):
        """Test creating a drum signal."""
        signal = Drum(sender="Marshal", recipient="Agent1", message="Attack!")
        assert signal.signal_type == SignalType.DRUM
        assert signal.sender == "Marshal"
        assert signal.recipient == "Agent1"
        assert signal.message == "Attack!"
    
    def test_create_gong_signal(self):
        """Test creating a gong signal."""
        signal = Gong(sender="Marshal", recipient="Agent1")
        assert signal.signal_type == SignalType.GONG
        assert signal.recipient == "Agent1"
    
    def test_drum_send(self):
        """Test sending a drum signal."""
        signal = Drum(recipient="Agent1", message="Advance!")
        result = signal.send()
        assert "🥁" in result
        assert "Agent1" in result
        assert "Advance!" in result
    
    def test_gong_send(self):
        """Test sending a gong signal."""
        signal = Gong(recipient="Agent1")
        result = signal.send()
        assert "🔔" in result
        assert "Agent1" in result
    
    def test_drum_helper_function(self):
        """Test drum helper function."""
        result = drum(sender="Marshal", recipient="Agent1", message="Go!")
        assert "🥁" in result
        assert "Agent1" in result
    
    def test_gong_helper_function(self):
        """Test gong helper function."""
        result = gong(sender="Marshal", recipient="Agent1")
        assert "🔔" in result
        assert "Agent1" in result
    
    def test_signal_str_representation(self):
        """Test string representation of signal.

        ★ 旧断言写的是 `"DRUM" in str_repr` —— 那是**基类** Signal.send()
          的格式（`🥁 DRUM signal sent to X`）。Drum 后来重写了 send()，
          改成了 `🥁 Drum to 'X'`，而测试没跟上。

        ★ 现在断言**语义**而不是字面大小写：
          一条信号的字符串形式必须能让人看出
          【这是什么信号】与【发给谁】—— 换个大小写不应该让它变红。
        """
        signal = Drum(recipient="TestAgent", message="Test")
        str_repr = str(signal)
        assert "drum" in str_repr.lower(), f"看不出这是击鼓信号：{str_repr}"
        assert "TestAgent" in str_repr
        assert "Test" in str_repr, "消息内容丢了"

    def test_repr_carries_the_signal_type(self):
        """★ `repr` 是调试时看到的东西，必须带类型与收件人。"""
        signal = Drum(sender="Marshal", recipient="TestAgent")
        text = repr(signal)
        assert "drum" in text.lower()
        assert "TestAgent" in text
