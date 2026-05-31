"""
BingFu 可视化样式常量
古代军事中军帐风格配色与字体
"""

# 主色调
COLORS = {
    # 背景色系
    "bg_dark": "#0a0e1a",          # 深蓝黑背景
    "bg_medium": "#111827",        # 中等深色
    "bg_light": "#1a2332",          # 浅深色
    "bg_highlight": "#243044",     # 高亮区域

    # 金色系（主要强调色）
    "gold_dark": "#8B6914",         # 暗金
    "gold": "#D4AF37",              # 主金色
    "gold_bright": "#FFD700",       # 亮金
    "gold_light": "#F5DEB3",        # 浅金/米色
    "gold_dim": "#8B6914",          # 暗金色（与gold_dark相同）

    # 文字色
    "text_primary": "#F5F5DC",       # 米白色主文字
    "text_secondary": "#C0C0C0",    # 灰色次要文字
    "text_muted": "#A8A8B8",        # 浅灰蓝 — 提升深色背景对比度

    # 状态色
    "status_online": "#50C878",     # 在线绿
    "status_busy": "#FFD700",       # 忙碌黄
    "status_offline": "#DC143C",    # 离线红
    "status_idle": "#6495ED",       # 空闲蓝

    # 边框色
    "border_gold": "#D4AF37",
    "border_gold_dim": "#8B6914",
}

# 字体配置
FONTS = {
    "title": ("Microsoft YaHei UI", 18, "bold"),       # 标题字体
    "subtitle": ("Microsoft YaHei UI", 13, "bold"),   # 副标题
    "body": ("Microsoft YaHei UI", 11),               # 正文
    "mono": ("Consolas", 11),                          # 等宽字体
    "small": ("Microsoft YaHei UI", 10),               # 小字
}

# 边框样式
BORDER_STYLE = {
    "width": 2,
    "relief": "groove",  # 立体边框效果
}

# 图标（使用Unicode古代符号）
ICONS = {
    "general": "🛡️",        # 将领
    "battle": "⚔️",          # 战役
    "report": "📜",         # 军情
    "drum": "🥁",           # 击鼓
    "bell": "🔔",           # 鸣金
    "arrow": "➤",          # 箭头
    "star": "★",            # 星号
    "dot": "●",              # 圆点
    "line": "═",            # 横线
    "corner": "◆",          # 角落装饰
}

# 装饰边框
BORDER_DOUBLE = "╔════════════════════════════════════════════════════╗"
BORDER_HORIZONTAL = "══════════════════════════════════════════════════"
BORDER_VERTICAL = "║"

# 中文数字映射
CHINESE_NUMBERS = {
    0: "零", 1: "一", 2: "二", 3: "三", 4: "四",
    5: "五", 6: "六", 7: "七", 8: "八", 9: "九",
    10: "十", 100: "百", 1000: "千", 10000: "万"
}

def to_chinese_number(n: int) -> str:
    """将阿拉伯数字转换为中文"""
    if n <= 10:
        return CHINESE_NUMBERS.get(n, str(n))
    elif n < 100:
        tens = n // 10
        ones = n % 10
        if ones == 0:
            return f"{CHINESE_NUMBERS[tens * 10]}"
        return f"{CHINESE_NUMBERS[tens * 10]}{CHINESE_NUMBERS[ones]}"
    return str(n)
