"""Tests for OpenAI-compatible tool / function calling."""

import json

import pytest

from llm_pipeline.production.serving import (
    ToolCall,
    format_tools_as_system_prompt,
    parse_tool_calls,
)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}


_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to one or more recipients",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}


# --------------------------------------------------------------------------- #
# System-prompt formatting
# --------------------------------------------------------------------------- #


def test_format_tools_includes_each_tool_name_and_signature():
    text = format_tools_as_system_prompt([_WEATHER_TOOL, _EMAIL_TOOL])
    assert "get_weather" in text
    assert "send_email" in text
    # Required fields appear without "?", optional with "?".
    assert "location: string" in text
    assert "unit?: string" in text
    assert "to: string" in text and "subject: string" in text


def test_format_tools_includes_description():
    text = format_tools_as_system_prompt([_WEATHER_TOOL])
    assert "Get current weather for a location" in text


def test_format_tools_calling_convention_is_present():
    text = format_tools_as_system_prompt([_WEATHER_TOOL])
    assert "<tool_call>" in text
    assert "</tool_call>" in text


def test_format_tools_empty_list_returns_empty_string():
    assert format_tools_as_system_prompt([]) == ""


def test_format_tools_skips_non_function_entries():
    text = format_tools_as_system_prompt([
        {"type": "code_interpreter"},                  # not "function" — skip
        _WEATHER_TOOL,
    ])
    assert "get_weather" in text
    # The non-function entry shouldn't bleed any garbage into the output.
    assert "code_interpreter" not in text


# --------------------------------------------------------------------------- #
# Output parsing — marker-based
# --------------------------------------------------------------------------- #


def test_parse_marker_based_tool_call():
    text = (
        "I'll check the weather for you.\n"
        '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>'
    )
    leftover, calls = parse_tool_calls(text)
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"location": "SF"}
    assert "I'll check the weather" in leftover
    assert "<tool_call>" not in leftover


def test_parse_multiple_tool_calls_in_one_response():
    text = (
        '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "f2", "arguments": {"x": 1}}</tool_call>'
    )
    _, calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["f1", "f2"]


def test_parse_tool_call_emits_uuid_id():
    text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    _, calls = parse_tool_calls(text)
    assert calls[0].id.startswith("call_")
    assert len(calls[0].id) > len("call_")             # has a UUID-ish suffix


def test_parse_no_tool_call_returns_empty():
    leftover, calls = parse_tool_calls("just regular text")
    assert calls == []
    assert leftover == "just regular text"


def test_parse_handles_json_decode_error_gracefully():
    text = '<tool_call>{not valid json}</tool_call>'
    leftover, calls = parse_tool_calls(text)
    assert calls == []                                 # bad payload skipped
    # The surrounding text still goes through (markers preserved since payload was bad).
    assert "<tool_call>" in leftover or leftover == ""


# --------------------------------------------------------------------------- #
# Output parsing — heuristic fallback
# --------------------------------------------------------------------------- #


def test_parse_falls_back_to_heuristic_for_raw_json():
    """Models that emit raw tool-call JSON (without markers) still parse."""
    text = '{"name": "get_weather", "arguments": {"location": "NYC"}}'
    _, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"


def test_to_dict_matches_openai_shape():
    text = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
    _, calls = parse_tool_calls(text)
    d = calls[0].to_dict()
    assert d["type"] == "function"
    assert d["function"]["name"] == "ping"
    assert "id" in d


def test_arguments_serialised_as_string_not_object():
    """Per OpenAI spec, ``function.arguments`` is a JSON-encoded string,
    NOT a parsed object. The parser keeps that shape.
    """
    text = '<tool_call>{"name": "x", "arguments": {"k": 1}}</tool_call>'
    _, calls = parse_tool_calls(text)
    assert isinstance(calls[0].arguments, str)
    assert json.loads(calls[0].arguments) == {"k": 1}
