"""OpenAI-compatible tool / function calling.

Two helpers:

- ``format_tools_as_system_prompt(tools)`` — turn a list of OpenAI
  ``ChatCompletionTool``-shaped dicts into a system-prompt suffix the
  model can be conditioned on. The suffix tells the model what tools
  are available and how to invoke them via a structured response.

- ``parse_tool_calls(text)`` — scan a model-generated string for
  tool-call markers (``<tool_call>...</tool_call>`` blocks) and parse
  them into the OpenAI ``message.tool_calls`` array format. Falls back
  to a heuristic JSON-extraction when the model doesn't use the markers
  but emits raw JSON.

The protocol matches OpenAI's chat-completions ``tool_choice`` semantics
(``"auto"`` vs ``"required"`` vs ``{"type": "function", "function": {"name": "X"}}``)
without depending on OpenAI's actual API — it's the format the model
sees on input and emits on output, formatted for the project's serving
layer.

Why marker-based, not JSON-only
-------------------------------

Models that haven't been fine-tuned for raw-JSON tool calls reliably
mix natural-language preamble with the structured call (e.g. "I'll
look up the weather. ``{"name": ...}``"). Parsing the JSON out of
arbitrary text with heuristics is brittle. A model trained to emit
``<tool_call>{...}</tool_call>`` blocks gives a clean parser and
unambiguous boundaries; the heuristic fallback handles models that
weren't trained that way.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Types — shaped to match OpenAI's chat-completions schema.
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    """Parsed tool invocation. ``arguments`` is a JSON string per OpenAI spec."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


# --------------------------------------------------------------------------- #
# System-prompt formatting
# --------------------------------------------------------------------------- #


def format_tools_as_system_prompt(
    tools: Sequence[Dict[str, Any]],
    tool_choice: Any = "auto",
) -> str:
    """Render tools + tool_choice as a system-prompt suffix.

    Output format (a stable convention this project uses; not OpenAI's
    internal one):

        # Available tools
        - get_weather(location: string, unit?: string): Get the current weather for a location
        - send_email(to: string, subject: string, body: string): Send an email

        # Calling convention
        Wrap any tool call in <tool_call>...</tool_call> with a JSON object:
            {"name": "<function-name>", "arguments": {<json args>}}

        # Tool selection
        tool_choice=auto       — call a tool only when needed
    """
    if not tools:
        return ""

    lines = ["# Available tools"]
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        name = fn.get("name", "<unnamed>")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        sig = _summarise_signature(params)
        lines.append(f"- {name}({sig}): {desc}".rstrip(": "))

    lines.append("")
    lines.append("# Calling convention")
    lines.append("Wrap any tool call in <tool_call>...</tool_call> with a JSON object:")
    lines.append('    {"name": "<function-name>", "arguments": {<json args>}}')
    lines.append("")
    lines.append("# Tool selection")
    lines.append(f"tool_choice={_format_tool_choice(tool_choice)}")
    return "\n".join(lines)


def _summarise_signature(params_schema: Dict[str, Any]) -> str:
    """Render a JSON schema's top-level properties as ``name: type`` pairs."""
    if not isinstance(params_schema, dict):
        return ""
    props = params_schema.get("properties", {}) or {}
    required = set(params_schema.get("required", []) or [])
    parts: List[str] = []
    for name, schema in props.items():
        ty = schema.get("type", "any") if isinstance(schema, dict) else "any"
        if name not in required:
            parts.append(f"{name}?: {ty}")
        else:
            parts.append(f"{name}: {ty}")
    return ", ".join(parts)


def _format_tool_choice(tool_choice: Any) -> str:
    if isinstance(tool_choice, str):
        descriptions = {
            "auto": "call a tool only when needed",
            "required": "must call exactly one tool",
            "none": "do not call any tool",
        }
        return f"{tool_choice}       — {descriptions.get(tool_choice, '?')}"
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        return f"{fn.get('name', '?')}     — must call this specific tool"
    return "auto"


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #


_MARKER_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _find_balanced_json_object(text: str, start: int) -> Optional[str]:
    """Starting at ``text[start]`` (which must be ``{``), scan forward and
    return the substring that balances braces. Handles nested objects and
    string-escaped braces. Returns None if no balanced match exists.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _find_tool_call_object(text: str) -> Optional[tuple[int, str]]:
    """Find the first ``{...}`` block that decodes as a JSON object with
    ``"name"`` and ``"arguments"`` keys. Returns ``(start_index, payload)``
    or None.
    """
    pos = 0
    while pos < len(text):
        next_brace = text.find("{", pos)
        if next_brace < 0:
            return None
        candidate = _find_balanced_json_object(text, next_brace)
        if candidate is not None:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                pos = next_brace + 1
                continue
            if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                return next_brace, candidate
        pos = next_brace + 1
    return None


def parse_tool_calls(text: str) -> tuple[str, List[ToolCall]]:
    """Extract tool calls from a model-generated string.

    Returns ``(remaining_text, tool_calls)`` where ``remaining_text`` is
    the input with the matched ``<tool_call>...</tool_call>`` blocks
    stripped. Useful for surfacing both the user-facing prose and the
    structured call to the API caller.

    Marker-based extraction is preferred. If no markers are present and
    the body looks like a JSON tool-call object, fall back to heuristic
    extraction.
    """
    calls: List[ToolCall] = []
    leftover = text

    # Marker-based: prefer this when present.
    matches = list(_MARKER_RE.finditer(text))
    if matches:
        for m in matches:
            try:
                payload = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            calls.append(_to_tool_call(payload))
        leftover = _MARKER_RE.sub("", text).strip()
        return leftover, calls

    # Heuristic fallback for un-marked output: balance-brace scan for the
    # first JSON object containing both ``name`` and ``arguments`` keys.
    found = _find_tool_call_object(text)
    if found is not None:
        start, payload_str = found
        try:
            payload = json.loads(payload_str)
            calls.append(_to_tool_call(payload))
            leftover = (text[:start] + text[start + len(payload_str):]).strip()
        except json.JSONDecodeError:
            pass

    return leftover, calls


def _to_tool_call(payload: Dict[str, Any]) -> ToolCall:
    name = payload.get("name", "<unnamed>")
    args = payload.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args)
    return ToolCall(id=f"call_{uuid.uuid4().hex[:16]}", name=name, arguments=args)
