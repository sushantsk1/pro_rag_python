import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path("data").resolve())
    port: int = 8787
    model_mode: str = "nvidia"
    embedding_mode: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: Path = field(default_factory=lambda: Path("data/models").resolve())
    nvidia_api_key: str = field(default="", repr=False)
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_timeout_ms: int = 30000
    nvidia_enable_thinking: bool = False
    upload_limit_bytes: int = 25 * 1024 * 1024
    document_limit: int = 100
    parse_timeout_seconds: float = 45

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        self.model_cache_dir = Path(self.model_cache_dir).resolve()
        # The original JS app used an ONNX export of the same model family.
        if self.embedding_model == "Xenova/all-MiniLM-L6-v2":
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        if self.model_mode not in {"nvidia", "stub"} or self.embedding_mode not in {"local", "stub"}:
            raise ValueError("MODEL_MODE must be nvidia/stub and EMBEDDING_MODE must be local/stub.")


def load_config() -> Config:
    for name in (".env", ".ENV"):
        if Path(name).is_file():
            load_dotenv(name, override=False)
            break
    env = os.environ
    return Config(
        data_dir=Path(env.get("DATA_DIR", "./data")),
        port=int(env.get("PORT", "8787")),
        model_mode=env.get("MODEL_MODE", "nvidia"),
        embedding_mode=env.get("EMBEDDING_MODE", "local"),
        embedding_model=env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_cache_dir=Path(env.get("MODEL_CACHE_DIR", "./data/models")),
        nvidia_api_key=env.get("NVIDIA_API_KEY", env.get("nvidia_api_key", "")),
        nvidia_model=env.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        nvidia_base_url=env.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        nvidia_timeout_ms=int(env.get("NVIDIA_TIMEOUT_MS", "30000")),
        nvidia_enable_thinking=env.get("NVIDIA_ENABLE_THINKING", "false").lower() == "true",
    )
