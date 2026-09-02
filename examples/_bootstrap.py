"""让 examples/ 下的脚本在**未安装包**的情况下也能直接跑。

════════════════════════════════════════════════════════════════
 ★ 这个文件补的是什么
════════════════════════════════════════════════════════════════

实测（2026-08-18）：clone 下来直接 `python examples/quickstart.py`，
**六个示例无一例外**都是：

    ModuleNotFoundError: No module named 'bingfu'

原因是脚本所在目录（examples/）会被自动加进 sys.path，
但**包根目录（bingfu-framework/）不会**。而这个包也没有被安装
—— egg-info 目录是旧构建残留，不代表装过。

★ 示例是一个框架的第一接触面。新人跑的第一条命令就崩，
  他不会去读 README 找原因，他会直接关掉。

★ 为什么不写「请先 pip install -e .」了事：
  那把「能不能跑」的责任推给了读者，而这三行就能让它无条件可跑。
  已安装时这段也不会有副作用 —— 包根排在 sys.path 前面，
  import 到的是同一份源码。

★ 集中在这里而不是每个示例各抄三行：抄 11 份的结果是
  改的时候漏掉几份，而漏掉的那几份**照样"看起来正常"**，
  直到有人恰好跑到它。
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
"""bingfu-framework/ —— `bingfu` 包所在的那一层。

★ 注意 launch.py 里曾经写成 `dirname(dirname(...))`，
  那指向的是 BingFuAgent/，**再上一层**，底下根本没有 bingfu 包。
  它没出事只是因为直接运行脚本时 Python 自动把脚本目录也加进了
  sys.path —— 也就是说那行 sys.path.insert 一直是无效代码。
"""

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
