from types import SimpleNamespace

import pytest

from src.config.model_configs import APIProvider, ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.model_client.openai_client import (
    _OpenAIStreamAccumulator,
    _build_reasoning_key,
    _convert_messages,
    _default_normal_response_parser,
    _inject_assistant_reasoning_content,
    _parse_tool_arguments,
    _sanitize_messages_for_toolless_request,
)
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType, TextMessagePart
from src.llm_models.payload_content.tool_option import ToolCall


@pytest.mark.parametrize("parse_mode", list(ToolArgumentParseMode))
def test_parse_tool_arguments_treats_blank_arguments_as_empty_dict(parse_mode: ToolArgumentParseMode) -> None:
    assert _parse_tool_arguments("", parse_mode, None) == {}
    assert _parse_tool_arguments("   ", parse_mode, None) == {}


def test_normal_response_parser_accepts_empty_string_arguments_for_parameterless_tool() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="finish-call",
                            type="function",
                            function=SimpleNamespace(name="finish", arguments=""),
                        )
                    ],
                ),
            )
        ],
        usage=None,
        model="glm-5.1",
    )

    api_response, usage_record = _default_normal_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=None,
    )

    assert len(api_response.tool_calls) == 1
    assert api_response.tool_calls[0].func_name == "finish"
    assert api_response.tool_calls[0].args == {}
    assert usage_record is None


def test_sanitize_messages_for_toolless_request_drops_assistant_tool_call_without_parts() -> None:
    messages = [
        Message(
            role=RoleType.Assistant,
            tool_calls=[
                ToolCall(
                    call_id="call_1",
                    func_name="mute_user",
                    args={"target": "alice"},
                )
            ],
        ),
        Message(
            role=RoleType.User,
            parts=[TextMessagePart(text="继续")],
        ),
    ]

    sanitized_messages = _sanitize_messages_for_toolless_request(messages)

    assert len(sanitized_messages) == 1
    assert sanitized_messages[0].role == RoleType.User


def test_normal_response_parser_ignores_reasoning_field_for_non_openrouter_provider() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="正式回复",
                    reasoning="推理内容",
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
        model="openrouter/test-model",
    )

    api_response, usage_record = _default_normal_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=_build_reasoning_key(
            APIProvider(name="test", base_url="https://openrouter.ai.example.com/api/v1", api_key="test")
        ),
    )

    assert api_response.content == "正式回复"
    assert api_response.reasoning_content is None
    assert usage_record is None


def test_normal_response_parser_reads_provider_reasoning_field_for_reasoning_domains() -> None:
    provider_urls = [
        "https://openrouter.ai/compatible-api",
        "https://api.groq.com/openai/v1",
    ]

    for provider_url in provider_urls:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="正式回复",
                        reasoning="推理内容",
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
            model="test-model",
        )

        api_response, usage_record = _default_normal_response_parser(
            response,
            reasoning_parse_mode=ReasoningParseMode.AUTO,
            tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
            reasoning_key=_build_reasoning_key(
                APIProvider(name="reasoning-provider", base_url=provider_url, api_key="test")
            ),
        )

        assert api_response.content == "正式回复"
        assert api_response.reasoning_content == "推理内容"
        assert usage_record is None


def test_inject_assistant_reasoning_content_only_fills_tool_call_spans() -> None:
    """有工具调用的 span 才回灌思维链；纯文本 span 的旧 ``reasoning_content`` 一并清掉。

    同时验证字段名跟 ``_build_reasoning_key`` 解析侧保持一致：OpenRouter 用 ``reasoning``。
    """

    messages = [
        MessageBuilder().set_role(RoleType.User).add_text_content("在吗").build(),
        MessageBuilder()
        .set_role(RoleType.Assistant)
        .add_text_content("**分析:** 直接回复")
        .set_tool_calls([ToolCall(call_id="call_1", func_name="reply", args={"msg_id": "1"})])
        .set_reasoning_content("planner 真实思维链")
        .build(),
        Message(role=RoleType.Tool, parts=[TextMessagePart(text="回复已发送。")], tool_call_id="call_1"),
        MessageBuilder().set_role(RoleType.Assistant).add_text_content("在的，咋啦").build(),
        MessageBuilder().set_role(RoleType.User).add_text_content("好的").build(),
        MessageBuilder()
        .set_role(RoleType.Assistant)
        .add_text_content("收到。")
        .set_reasoning_content("不需要回灌的思维链")
        .build(),
    ]
    converted_messages = _convert_messages(messages)
    openrouter_key = _build_reasoning_key(
        APIProvider(name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="test")
    )

    _inject_assistant_reasoning_content(converted_messages, reasoning_key=openrouter_key)
    assistant_payloads = [item for item in converted_messages if item["role"] == "assistant"]
    assert openrouter_key == "reasoning"
    # 工具调用 span：planner + guided_reply 都补上了真实思维链
    assert assistant_payloads[0]["reasoning"] == "planner 真实思维链"
    assert assistant_payloads[1]["reasoning"] == "planner 真实思维链"
    # 纯文本 span：不注入新字段，旧 reasoning_content 也被清理
    assert "reasoning" not in assistant_payloads[2]
    # 内部统一字段名在所有 assistant 上都不应残留
    assert all("reasoning_content" not in payload for payload in assistant_payloads)


def test_stream_accumulator_reads_openrouter_reasoning_delta_field() -> None:
    accumulator = _OpenAIStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        reasoning_key=_build_reasoning_key(
            APIProvider(name="openrouter", base_url="https://openrouter.ai/compatible-api", api_key="test")
        ),
    )
    try:
        accumulator.process_delta(SimpleNamespace(reasoning="流式推理", content=None, tool_calls=None))
        accumulator.process_delta(SimpleNamespace(content="正式回复", tool_calls=None))

        api_response = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "正式回复"
    assert api_response.reasoning_content == "流式推理"
