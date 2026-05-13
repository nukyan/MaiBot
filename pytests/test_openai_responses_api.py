"""OpenAI Responses API（``/v1/responses``）适配的核心单元测试。"""

from types import SimpleNamespace

import pytest

from src.config.model_configs import ModelInfo, ReasoningParseMode, ToolArgumentParseMode, WireApi
from src.llm_models.openai_responses import (
    _ResponsesStreamAccumulator,
    convert_messages_to_response_input,
    convert_response_format,
    convert_tool_options,
    extract_usage_record,
    parse_response,
)
from src.llm_models.payload_content.message import Message, RoleType, TextMessagePart
from src.llm_models.payload_content.resp_format import JsonSchema, RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption


def test_model_info_wire_api_normalization() -> None:
    """覆盖默认值、大小写归一化、非法值校验三种情形。"""
    assert ModelInfo(model_identifier="m", name="n", api_provider="p").wire_api == WireApi.CHAT.value

    normalized = ModelInfo(
        model_identifier="m", name="n", api_provider="p", wire_api="RESPONSES"
    )
    assert normalized.wire_api == WireApi.RESPONSES.value

    with pytest.raises(ValueError, match="wire_api"):
        ModelInfo(model_identifier="m", name="n", api_provider="p", wire_api="grpc")


def test_convert_messages_to_response_input_handles_all_roles() -> None:
    messages = [
        Message(role=RoleType.System, parts=[TextMessagePart(text="你是助手")]),
        Message(role=RoleType.User, parts=[TextMessagePart(text="你好")]),
        Message(
            role=RoleType.Assistant,
            parts=[TextMessagePart(text="正在调用工具")],
            tool_calls=[ToolCall(call_id="call-1", func_name="search", args={"q": "今日天气"})],
        ),
        Message(
            role=RoleType.Tool,
            parts=[TextMessagePart(text="多云转晴")],
            tool_call_id="call-1",
            tool_name="search",
        ),
    ]

    converted = convert_messages_to_response_input(messages)

    assert converted[0] == {"role": "system", "content": [{"type": "input_text", "text": "你是助手"}]}
    assert converted[1] == {"role": "user", "content": [{"type": "input_text", "text": "你好"}]}
    assert converted[2] == {"role": "assistant", "content": "正在调用工具"}
    assert converted[3] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "search",
        "arguments": '{"q": "今日天气"}',
    }
    assert converted[4] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "多云转晴",
    }


def test_convert_tool_options_uses_flat_function_schema() -> None:
    tools = convert_tool_options(
        [
            ToolOption(
                name="lookup_user",
                description="查询用户",
                parameters_schema_override={
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            )
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "lookup_user",
            "description": "查询用户",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            "strict": False,
        }
    ]


def test_convert_response_format_covers_text_json_object_and_schema() -> None:
    assert convert_response_format(None) is None
    assert convert_response_format(RespFormat()) is None
    assert convert_response_format(RespFormat(format_type=RespFormatType.JSON_OBJ)) == {
        "format": {"type": "json_object"}
    }

    json_schema: JsonSchema = {
        "name": "ReplySchema",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        "strict": True,
    }
    result = convert_response_format(RespFormat(format_type=RespFormatType.JSON_SCHEMA, schema=json_schema))
    assert result is not None
    assert result["format"] == {
        "type": "json_schema",
        "name": "ReplySchema",
        "schema": json_schema["schema"],
        "strict": True,
    }


def test_extract_usage_record_includes_cached_tokens() -> None:
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=50,
        total_tokens=170,
        input_tokens_details=SimpleNamespace(cached_tokens=20),
        output_tokens_details=SimpleNamespace(reasoning_tokens=10),
    )
    assert extract_usage_record(usage) == (120, 50, 170, 20, 100)
    assert extract_usage_record(None) is None


def test_parse_response_extracts_text_reasoning_and_tool_call() -> None:
    response = SimpleNamespace(
        model="gpt-5",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[SimpleNamespace(type="reasoning_text", text="思考过程")],
                summary=[SimpleNamespace(type="summary_text", text="一句话总结")],
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="你好"),
                    SimpleNamespace(type="output_text", text="，世界"),
                ],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-77",
                name="search",
                arguments='{"q": "今日天气"}',
            ),
        ],
    )

    api_response, usage_tuple = parse_response(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    assert api_response.content == "你好，世界"
    assert api_response.reasoning_content == "思考过程一句话总结"
    assert api_response.tool_calls is not None
    assert api_response.tool_calls[0].call_id == "call-77"
    assert api_response.tool_calls[0].func_name == "search"
    assert api_response.tool_calls[0].args == {"q": "今日天气"}
    assert usage_tuple == (12, 8, 20, 2, 10)


def test_stream_accumulator_handles_text_reasoning_and_function_call_events() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )
    accumulator.process_event(SimpleNamespace(type="response.output_text.delta", delta="第一段"))
    accumulator.process_event(SimpleNamespace(type="response.output_text.delta", delta="第二段"))
    accumulator.process_event(SimpleNamespace(type="response.reasoning_text.delta", delta="思考"))
    accumulator.process_event(SimpleNamespace(type="response.reasoning_summary_text.delta", delta="总结"))
    accumulator.process_event(
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(type="function_call", call_id="call-stream", name="lookup"),
        )
    )
    accumulator.process_event(
        SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=0, delta='{"qu'
        )
    )
    accumulator.process_event(
        SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=0, delta='ery": "x"}'
        )
    )
    accumulator.process_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                model="gpt-5",
                usage=SimpleNamespace(
                    input_tokens=8,
                    output_tokens=12,
                    total_tokens=20,
                    input_tokens_details=SimpleNamespace(cached_tokens=4),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=2),
                ),
                incomplete_details=None,
            ),
        )
    )

    api_response = accumulator.build_response()

    assert api_response.content == "第一段第二段"
    assert api_response.reasoning_content == "思考总结"
    assert api_response.tool_calls is not None
    assert api_response.tool_calls[0].call_id == "call-stream"
    assert api_response.tool_calls[0].func_name == "lookup"
    assert api_response.tool_calls[0].args == {"query": "x"}
    assert accumulator.usage_record == (8, 12, 20, 4, 4)
