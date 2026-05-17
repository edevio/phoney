from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["CG", "OR"]


class Review(BaseModel):
    row_id: int = Field(ge=0)
    category: str
    rating: float
    label: Label
    text: str


class ProviderResponse(BaseModel):
    text: str
    latency_ms: int = Field(ge=0)
