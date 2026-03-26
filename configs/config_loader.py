import os
import yaml
from typing import Any, Dict


class Config:
    _data: Dict[str, Any] | None = None

    @classmethod
    def load(cls, config_path: str | None = None) -> Dict[str, Any]:
        if cls._data is None:
            path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(path, "r", encoding="utf-8") as f:
                cls._data = yaml.safe_load(f)
        return cls._data # type: ignore

    @classmethod
    def get(cls, key_path: str, default: Any | None = None) -> Any:
        data = cls.load()
        node: Any = data
        for key in key_path.split('.'):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node


config = Config.load()

# if __name__ == "__main__":
#     print(config)
