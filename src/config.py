# src/config.py
from pathlib import Path
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")

    llm_model: str = Field(default="gpt-4")
    embedding_model: str = Field(default="text-embedding-ada-002")
    openai_vision_model: str = Field(default="gpt-4o")

    retrieval_top_k: int = Field(default=5)
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    use_proxy: bool = Field(default=False)
    proxy_host: str = Field(default="")
    proxy_port_http: int = Field(default=0)
    proxy_port_socks5: int = Field(default=0)
    proxy_user: str = Field(default="")
    proxy_pass: str = Field(default="")

    @property
    def proxy_url_http(self) -> str:
        if not self.proxy_host or not self.proxy_port_http:
            return ""
        if self.proxy_user and self.proxy_pass:
            return f"http://{self.proxy_user}:{self.proxy_pass}@{self.proxy_host}:{self.proxy_port_http}"
        return f"http://{self.proxy_host}:{self.proxy_port_http}"

    @property
    def proxy_url_socks5(self) -> str:
        if not self.proxy_host or not self.proxy_port_socks5:
            return ""
        if self.proxy_user and self.proxy_pass:
            return f"socks5://{self.proxy_user}:{self.proxy_pass}@{self.proxy_host}:{self.proxy_port_socks5}"
        return f"socks5://{self.proxy_host}:{self.proxy_port_socks5}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


BASE_DIR = Path(__file__).parent.parent

def _resolve_data_dir() -> Path:
    raw = os.environ.get("DATA_DIR", "data")
    p = Path(raw)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

DATA_DIR = _resolve_data_dir()
PDFS_DIR = DATA_DIR / "pdfs"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

for dir_path in [DATA_DIR, PDFS_DIR, UPLOADS_DIR, PROCESSED_DIR, IMAGES_DIR, INDEX_DIR, EVAL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
