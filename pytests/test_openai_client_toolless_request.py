from types import SimpleNamespace

import pytest

from src.config.model_configs import APIProvider, ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.model_client.base_client import APIResponse
from src.llm_models.model_client.openai_client import (
    _OpenAIStreamAccumulator,
    _build_reasoning_key,
    _default_normal_response_parser,
    _parse_tool_arguments,
    _sanitize_messages_for_toolless_request,
    OpenaiClient,
)
from src.llm_models.payload_content.message import Message, RoleType, TextMessagePart
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

    converted_messages = [
        {"role": "user", "content": "在吗"},
        {
            "role": "assistant",
            "content": "**分析:** 直接回复",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "reply", "arguments": "{}"}}
            ],
            "reasoning_content": "planner 真实思维链",
        },
        {"role": "tool", "content": "回复已发送。", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "在的，咋啦"},
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "收到。", "reasoning_content": "不需要回灌的思维链"},
    ]
    client = OpenaiClient(
        APIProvider(name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="test")
    )

    client._inject_assistant_reasoning_content(converted_messages)
    assistant_payloads = [item for item in converted_messages if item["role"] == "assistant"]

    assert client.reasoning_key == "reasoning"
    # 工具调用 span：planner + 后续无工具 assistant 都补上了真实思维链
    assert assistant_payloads[0]["reasoning"] == "planner 真实思维链"
    assert assistant_payloads[1]["reasoning"] == "planner 真实思维链"
    # 纯文本 span：不注入新字段，旧 reasoning_content 也被清理
    assert "reasoning" not in assistant_payloads[2]
    # 内部统一字段名在所有 assistant 上都不应残留
    assert all("reasoning_content" not in payload for payload in assistant_payloads)


def test_openai_client_caches_and_hydrates_reasoning_content_by_tool_call_id() -> None:
    """客户端记录响应中的 ``reasoning_content``，下一轮自动按 ``tool_call_id`` 补回。"""

    client = OpenaiClient(
        APIProvider(name="DeepSeek", base_url="https://api.deepseek.com", api_key="test")
    )

    # 一次响应：reasoning_content + tool_calls 写入缓存
    api_response = APIResponse()
    api_response.reasoning_content = "上一轮思维链"
    api_response.tool_calls = [ToolCall(call_id="call_xyz", func_name="reply", args={})]
    client._remember_assistant_reasoning_content(api_response)

    assert client._reasoning_content_by_tool_call_id["call_xyz"] == "上一轮思维链"

    # 下一轮请求历史里没带 reasoning_content，hydrate 应按 id 自动补回
    converted_messages = [
        {
            "role": "assistant",
            "content": "**分析**",
            "tool_calls": [
                {"id": "call_xyz", "type": "function", "function": {"name": "reply", "arguments": "{}"}}
            ],
        },
    ]
    client._hydrate_reasoning_content_from_cache(converted_messages)

    assert converted_messages[0]["reasoning_content"] == "上一轮思维链"


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
