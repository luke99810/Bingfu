"""
Memory module (记忆/军需库模块)
Defines the Memory class — stores an agent's knowledge and experience.
Inspired by ancient Chinese warfare "军需库" (army supply).
"""

from typing import Any, Optional, Dict
from pydantic import BaseModel, Field, PrivateAttr
import json
import os


class Memory(BaseModel):
    """
    Memory (记忆) — stores an agent's knowledge and experience.
    Can be file-based or in-memory.
    """

    name: str = Field(default="default_memory", description="Memory name")
    memory_type: str = Field(
        default="file",
        description="Memory type: 'file' or 'dict' (in-memory)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="File path for file-based memory (used when memory_type='file')"
    )

    # 使用 PrivateAttr 存储内部数据
    _internal_data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize data from file if file-based
        if self.memory_type == "file" and self.file_path:
            self._load()

    def _load(self) -> None:
        """Load memory from file (if file-based)."""
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._internal_data = json.load(f)
            except Exception:
                self._internal_data = {}

    def _save(self) -> None:
        """Save memory to file (if file-based)."""
        if self.memory_type == "file" and self.file_path:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._internal_data, f, ensure_ascii=False, indent=2)

    def store(self, key: str, value: Any) -> None:
        """
        Store a key-value pair in memory.

        Args:
            key (str): The key.
            value (Any): The value.
        """
        self._internal_data[key] = value
        if self.memory_type == "file":
            self._save()

    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from memory.

        Args:
            key (str): The key.
            default (Any): Default value if key not found.

        Returns:
            Any: The stored value, or default.
        """
        return self._internal_data.get(key, default)

    def delete(self, key: str) -> bool:
        """
        Delete a key from memory.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        if key in self._internal_data:
            del self._internal_data[key]
            if self.memory_type == "file":
                self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear all memory."""
        self._internal_data.clear()
        if self.memory_type == "file" and self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

    def all(self) -> Dict[str, Any]:
        """
        Get all stored data.

        Returns:
            Dict[str, Any]: All key-value pairs.
        """
        return self._internal_data.copy()

    def __len__(self) -> int:
        return len(self._internal_data)

    def __str__(self) -> str:
        return f"Memory(name='{self.name}', type='{self.memory_type}', entries={len(self)})"

    def __repr__(self) -> str:
        return self.__str__()
