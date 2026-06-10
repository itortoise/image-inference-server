"""API 请求/响应模型。"""

from typing import Any, List

from pydantic import BaseModel, Field


class InferInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str = "BYTES"
    data: List[Any]


class InferRequest(BaseModel):
    inputs: List[InferInput]


class InnerMessage(BaseModel):
    """result.data.message 内层结构。"""
    code: str = "1"
    data: Any = None


class ResultData(BaseModel):
    """result.data 结构。"""
    message: InnerMessage


class SuccessResult(BaseModel):
    """成功时的 result 结构。"""
    code: str = "200"
    data: ResultData
    message: str = "success"


class ErrorResult(BaseModel):
    """错误时的 result 结构。"""
    code: str
    data: Any = None
    message: str


class UnifiedResponse(BaseModel):
    """统一响应格式。

    成功:
    {
        "status": true,
        "result": {
            "code": "200",
            "data": {"message": {"code": "1", "data": ...}},
            "message": "success"
        },
        "message": ""
    }

    错误:
    {
        "status": false,
        "result": {"code": "404", "data": null, "message": "..."},
        "message": ""
    }
    """
    status: bool
    result: Any
    message: str = ""


class ModelInfo(BaseModel):
    name: str
    version: str = "1"
    state: str = "READY"


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: dict[str, str]
