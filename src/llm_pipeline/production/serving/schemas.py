"""OpenAI-compatible request/response schemas (subset).

Pydantic v2 is required (declared in the ``[serving]`` extra). This module
is only imported by ``server.build_app()``, so importing the package without
the extra still works.
"""

from typing import List, Optional, Union

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "llm-pipeline"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
