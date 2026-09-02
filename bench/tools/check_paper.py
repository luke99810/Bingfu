# -*- coding: utf-8 -*-
"""论文自检：交叉引用、公式编号、文献引用闭合。"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\宿心\Desktop\兵符论文-改稿.md"
s = io.open(P, encoding="utf-8").read()

problems = 0


def bad(m):
    global problems
    problems += 1
    print("  x " + m)


# 小节交叉引用
heads = set(re.findall(r"^#{2,4} (\d+(?:\.\d+)*)", s, re.M))
for r in sorted(set(re.findall(r"§(\d+(?:\.\d+)*)", s))):
    if r not in heads:
        bad("引用了不存在的小节 §%s" % r)

# 公式编号连续
tags = [int(x) for x in re.findall(r"\\tag\{(\d+)\}", s)]
if tags != list(range(1, len(tags) + 1)):
    bad("公式编号不连续：%s" % tags)
else:
    print("  公式 (1)–(%d) 编号连续" % len(tags))
atags = re.findall(r"\\tag\{(A\.\d+)\}", s)
print("  附录公式：%s" % (atags or "无"))

# 正文中被引用的公式号必须存在
for n in set(re.findall(r"式 \((\d+)\)", s)):
    if int(n) not in tags:
        bad("正文引用了不存在的公式 (%s)" % n)

# 文献：正文引用 vs 文献表
body_end = s.index("## 参考文献")
body, bib = s[:body_end], s[body_end:]
cited = {int(x) for x in re.findall(r"\[(\d{1,2})\]", body)}
listed = {int(x) for x in re.findall(r"^\[(\d{1,2})\] ", bib, re.M)}
for x in sorted(cited - listed):
    bad("正文引用 [%d] 但文献表未列出" % x)
for x in sorted(listed - cited):
    bad("文献表列出 [%d] 但正文从未引用" % x)
print("  正文引用 %d 篇，文献表 %d 篇" % (len(cited), len(listed)))

# 表编号连续
tbls = [int(x) for x in re.findall(r"\*\*表 (\d+)　", s)]
if tbls != sorted(tbls) or tbls != list(range(1, len(tbls) + 1)):
    bad("表编号不连续：%s" % tbls)
else:
    print("  表 1–%d 编号连续" % len(tbls))

# 数字一致性：几个关键量在全文应当只有一个取值
for pat, want in ((r"每得分点 token 由 (\d+) 降至 (\d+)", ("1596", "955")),):
    for m in re.finditer(pat, s):
        if m.groups() != want:
            bad("关键数字不一致：%s" % (m.group(0),))

print("\n%s" % ("通过：未发现问题" if problems == 0 else "发现 %d 处问题" % problems))
sys.exit(0 if problems == 0 else 1)
