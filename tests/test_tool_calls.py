import json
import types

import pytest

from utils.tool_calls import ToolCallStream, parse_tool_calls, to_template_messages


def test_parse_hermes_blocks_with_surrounding_text():
    text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me check."
    assert [call["function"]["name"] for call in calls] == ["get_weather", "get_time"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert all(call["type"] == "function" and call["id"].startswith("call_") for call in calls)
    assert len({call["id"] for call in calls}) == 2


def test_parse_llama_json_reply_uses_parameters_key():
    content, calls = parse_tool_calls('{"name": "search", "parameters": {"query": "tides"}}')
    assert content == "" and calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "tides"}


@pytest.mark.parametrize("text", ["Just an answer.", "{not json}", '<tool_call>{"oops": 1}</tool_call>', '<tool_call>{"name": "x"', '{"a": 1}'])
def test_text_that_is_not_a_call_is_left_alone(text):
    content, calls = parse_tool_calls(text)
    assert calls == [] and content == text


def collect(pieces):
    stream = ToolCallStream()
    emitted = [stream.feed(piece) for piece in pieces]
    rest, calls = stream.finish()
    return "".join(emitted), rest, calls, emitted


def test_stream_passes_plain_text_through_immediately():
    text, rest, calls, emitted = collect(["Hel", "lo ", "there"])
    assert text == "Hello there" and rest == "" and calls == [] and emitted == ["Hel", "lo ", "there"]


def test_stream_holds_back_a_tag_split_across_pieces():
    pieces = ["Sure. <tool", "_call>", '{"name": "f", ', '"arguments": {"a": 1}}', "</tool_call>"]
    text, rest, calls, _ = collect(pieces)
    assert text == "Sure. " and rest == "" and len(calls) == 1 and calls[0]["function"]["name"] == "f"


def test_stream_releases_a_false_alarm_prefix():
    text, rest, calls, _ = collect(["a <", "b and <tool", " plan"])
    assert (text + rest) == "a <b and <tool plan" and calls == []


def test_stream_holds_json_reply_then_parses_or_releases_it():
    text, rest, calls, _ = collect(['{"name": "f", ', '"parameters": {}}'])
    assert text == "" and rest == "" and calls[0]["function"]["name"] == "f"
    text, rest, calls, _ = collect(['{"just": ', '"data"}'])
    assert text == "" and rest == '{"just": "data"}' and calls == []


def test_template_messages_parse_argument_strings():
    messages = [types.SimpleNamespace(role="assistant", content=None, name=None, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]),
                types.SimpleNamespace(role="tool", content="42", name="f", tool_calls=None)]
    converted = to_template_messages(messages)
    assert converted[0]["content"] == "" and converted[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert converted[1] == {"role": "tool", "content": "42", "name": "f"}
