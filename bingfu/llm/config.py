"""
LLM 配置管理 (兵符 · 军师调度配置)

管理多个 LLM provider 的配置，支持热切换默认模型。
配置来源：YAML 文件 / 代码创建 / 环境变量
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """把 `.env` 载进环境变量，整个进程只做一次。

    ★ 已存在的环境变量**优先**（python-dotenv 默认 override=False）：
      显式 `set DEEPSEEK_API_KEY=...` 应该能盖过项目里的 .env，
      否则临时换 key 调试会得到一个和你以为的不一样的结果。

    ★ 没装 python-dotenv 不是错误 —— 它是 optional-dependencies 里的东西。
      直接用环境变量的人不该被强迫装它。静默跳过即可。

    ★ 不在 import 时执行：一个库在被 import 的瞬间就去读磁盘、
      改进程环境，是很难排查的副作用。放在真正要取 key 的那一刻。
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)


class LLMConfig(BaseModel):
    """
    单个 LLM Provider 的配置 (一员军师的委任状)

    Attributes:
        provider: 提供商名称 (deepseek/openai/openai_compatible)
        api_key: API 密钥
        base_url: API 基础 URL（OpenAI 兼容接口可自定义）
        model: 模型名称
        temperature: 默认生成温度
        max_tokens: 默认最大 token 数
        extra: 额外参数（各 provider 特有的配置）
    """
    provider: str = Field(..., description="提供商: deepseek/openai/openai_compatible")
    api_key: str = Field(default="", description="API 密钥")
    base_url: Optional[str] = Field(default=None, description="API 基础 URL")
    model: str = Field(default="", description="模型名称")
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: int = Field(default=2048, description="最大生成 token 数")
    extra: Dict[str, Any] = Field(default_factory=dict, description="额外参数")

    def resolve_api_key(self) -> str:
        """
        解析 API 密钥

        优先级：直接配置 > 环境变量

        环境变量命名规则：
          - DeepSeek: DEEPSEEK_API_KEY
          - OpenAI: OPENAI_API_KEY
          - 自定义: LLM_API_KEY_{provider大写}
        """
        if self.api_key:
            return self.api_key

        # ★ 先把 .env 载进环境，再读环境变量。
        #
        #   此前这一步不存在：README 明确告诉用户「把 key 放进 .env」，
        #   而**主链路从来不读那个文件** —— load_dotenv() 只出现在
        #   langchain_integration.py 和一个示例里。
        #
        #   于是用户照文档配好 .env，launch.py 依然打印
        #   「⚠️ 未检测到 DEEPSEEK_API_KEY，将领将以关键词匹配模式运作」——
        #   一个**产出方齐全、消费方缺席**的缺口，
        #   而现象（降级运行）看起来像是"没配 key"，指不到真因。
        _load_dotenv_once()

        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_key = env_map.get(self.provider, f"LLM_API_KEY_{self.provider.upper()}")
        return os.environ.get(env_key, "")


class LLMManager(BaseModel):
    """
    LLM 配置管理器 (军师调度府)

    管理多个 LLM provider 配置，支持默认 provider 切换。
    """
    default_provider: str = Field(default="deepseek", description="默认使用的 provider")
    providers: Dict[str, LLMConfig] = Field(
        default_factory=dict,
        description="所有 provider 配置: {name: LLMConfig}"
    )

    class Config:
        arbitrary_types_allowed = True

    def add_provider(self, name: str, config: LLMConfig) -> None:
        """添加一个 provider 配置"""
        self.providers[name] = config

    def remove_provider(self, name: str) -> bool:
        """移除一个 provider 配置"""
        if name in self.providers:
            del self.providers[name]
            if self.default_provider == name:
                # 如果删除的是默认的，切换到第一个可用的
                if self.providers:
                    self.default_provider = next(iter(self.providers.keys()))
                else:
                    self.default_provider = ""
            return True
        return False

    def get_provider_config(self, name: Optional[str] = None) -> Optional[LLMConfig]:
        """
        获取指定 provider 的配置

        Args:
            name: provider 名称，为 None 时使用默认

        Returns:
            LLMConfig 或 None
        """
        key = name or self.default_provider
        return self.providers.get(key)

    def set_default(self, name: str) -> bool:
        """设置默认 provider"""
        if name in self.providers:
            self.default_provider = name
            return True
        return False

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "LLMManager":
        """
        从 YAML 配置字典创建 LLMManager

        期望格式:
        ```yaml
        llm:
          default_provider: deepseek
          providers:
            deepseek:
              provider: deepseek
              api_key: sk-xxx
              model: deepseek-chat
            openai:
              provider: openai
              api_key: sk-xxx
              model: gpt-4o
        ```
        """
        llm_data = data.get("llm", {})
        manager = cls(
            default_provider=llm_data.get("default_provider", "deepseek")
        )

        for name, cfg in llm_data.get("providers", {}).items():
            manager.add_provider(name, LLMConfig(**cfg))

        return manager

    def to_yaml_dict(self) -> Dict[str, Any]:
        """导出为 YAML 配置字典"""
        providers_dict = {}
        for name, cfg in self.providers.items():
            providers_dict[name] = cfg.model_dump(exclude_none=True)

        return {
            "llm": {
                "default_provider": self.default_provider,
                "providers": providers_dict
            }
        }

    def __str__(self) -> str:
        names = ", ".join(self.providers.keys())
        return f"LLMManager(default={self.default_provider}, providers=[{names}])"
