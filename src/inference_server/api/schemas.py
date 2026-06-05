"""API 请求/响应模型。"""

from typing import Any, List

from pydantic import BaseModel, Field


class InferInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str = "BYTES"
    data: List[Any]


class InferOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[Any]


class InferRequest(BaseModel):
    inputs: List[InferInput]


class InferResponse(BaseModel):
    model_name: str
    outputs: List[InferOutput]


class ModelInfo(BaseModel):
    name: str
    version: str = "1"
    state: str = "READY"


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: dict[str, str]
