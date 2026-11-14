"""Tool (function) calling for the chat server.

The model writes a call as text in whatever format its chat template taught it; this module turns that text into OpenAI-style
`tool_calls`, and converts OpenAI-style conversation messages into what the chat templates expect.

Supported output formats:
  * Hermes / Qwen:  <tool_call>{"name": "...", "arguments": {...}}</tool_call>   (any number of blocks, optional text around them)
  * Llama 3.x JSON: a whole reply that is one JSON object {"name": "...", "parameters": {...}}
"""
import json
import re
import uuid

OPEN_TAG = "<tool_call>"
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def make_call(name, arguments):
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}


def _call_from_object(obj):
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        arguments = obj.get("arguments", obj.get("parameters", {}))
        return make_call(obj["name"], arguments)
    return None


def parse_tool_calls(text):
    """Returns (text with the call blocks removed, [OpenAI tool_call dicts])."""
    calls = []

    def replace(match):
        try:
            call = _call_from_object(json.loads(match.group(1)))
        except json.JSONDecodeError:
            call = None
        if call is None:
            return match.group(0)  # not a call after all: leave it in the text
        calls.append(call)
        return ""

    remaining = TOOL_CALL_BLOCK.sub(replace, text)
    if not calls:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call = _call_from_object(json.loads(stripped))
            except json.JSONDecodeError:
                call = None
            if call is not None:
                return "", [call]
    return remaining.strip() if calls else remaining, calls


class ToolCallStream:
    """Passes ordinary text through as it streams but holds back anything that might be a tool call.

    feed(piece) returns the text that is safe to send now; finish() returns (text still owed, tool calls). A reply that
    starts with "{" is held back entirely (it may be a Llama-style JSON call); a "<tool_call>" tag, or the start of one at
    the end of the buffer, is held back until the reply ends.
    """

    def __init__(self):
        self.full = ""
        self.pending = ""
        self.emitted = ""
        self.in_call = False

    def feed(self, piece):
        self.full += piece
        if self.in_call:
            return ""
        self.pending += piece
        index = self.pending.find(OPEN_TAG)
        if index >= 0:
            out, self.pending, self.in_call = self.pending[:index], self.pending[index:], True
        elif not self.emitted and self.pending.lstrip().startswith("{"):
            return ""
        else:
            keep = 0
            for length in range(min(len(OPEN_TAG) - 1, len(self.pending)), 0, -1):
                if self.pending.endswith(OPEN_TAG[:length]):
                    keep = length
                    break
            out, self.pending = self.pending[:len(self.pending) - keep], self.pending[len(self.pending) - keep:]
        self.emitted += out
        return out

    def finish(self):
        content, calls = parse_tool_calls(self.full)
        rest = content[len(self.emitted):] if content.startswith(self.emitted) else ""
        return rest, calls


def to_template_messages(messages):
    """OpenAI-style messages -> the dicts chat templates expect (tool-call arguments as objects, not JSON strings)."""
    converted = []
    for message in messages:
        entry = {"role": message.role, "content": message.content if message.content is not None else ""}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = []
            for call in tool_calls:
                function = dict(call.get("function", {}))
                if isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"])
                    except json.JSONDecodeError:
                        pass
                entry["tool_calls"].append({"type": "function", "function": function})
        if getattr(message, "name", None):
            entry["name"] = message.name
        converted.append(entry)
    return converted
