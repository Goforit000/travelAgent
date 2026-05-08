"""
配置管理 — 整个项目的唯一配置入口

为什么单独一个文件？
- 所有层（tools / graph / api）都需要读配置
- 集中管理意味着改一个地方就够了
- pydantic-settings 自动从 .env 文件和环境变量读取
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """应用配置，字段名自动匹配环境变量（不区分大小写）"""

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # 高德地图
    amap_api_key: str = ""

    # Unsplash
    unsplash_access_key: str = ""

    # 服务
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


# 全局单例，import 即用
settings = Settings()