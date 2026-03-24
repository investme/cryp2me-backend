from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000"

    binance_api_key: str = ""
    binance_secret:  str = ""
    coingecko_api_key: str = ""

    redis_url: str = "redis://localhost:6379"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
