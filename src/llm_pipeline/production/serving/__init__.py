"""HTTP serving layer."""

from .server import build_app, run
from .lora_router import LoRARouter
from .tool_calling import (
    ToolCall,
    format_tools_as_system_prompt,
    parse_tool_calls,
)
from .auth import (
    APIKey,
    APIKeyStore,
    AuthError,
    InMemoryAPIKeyStore,
    JSONFileAPIKeyStore,
    TokenBucketLimiter,
    verify_api_key,
)
from .multi_model import (
    ModelEntry,
    MultiModelRegistry,
    select_engine_for_request,
)
from .queue import (
    BackpressureConfig,
    BackpressurePolicy,
    Priority,
    PriorityRequestQueue,
    QueueOverflow,
)

__all__ = [
    "build_app", "run",
    "LoRARouter",
    "ToolCall",
    "format_tools_as_system_prompt",
    "parse_tool_calls",
    "APIKey", "APIKeyStore", "AuthError",
    "InMemoryAPIKeyStore", "JSONFileAPIKeyStore",
    "TokenBucketLimiter", "verify_api_key",
    "ModelEntry", "MultiModelRegistry", "select_engine_for_request",
    "BackpressureConfig", "BackpressurePolicy",
    "Priority", "PriorityRequestQueue", "QueueOverflow",
]
