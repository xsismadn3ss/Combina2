from dataclasses import dataclass

@dataclass(frozen=True)
class APICOnfig:
    prefix: str = "/api/v1"
