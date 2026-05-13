"""OpenAI Responses API（``/v1/responses``）适配辅助。

本模块承担「内部统一对象 <-> Responses API 数据结构」的纯函数转换以及流式累积逻辑，
具体的 SDK 调用、错误处理、快照保存仍由 ``openai_client`` 统一编排。
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast
from uuid import uuid4

from openai import AsyncStream
from openai.types.responses import (
    Response,
    ResponseInputContentParam,
    ResponseInputParam,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ToolParam,
)

from src.common.logger import get_logger
from src.config.model_configs import ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.exceptions import EmptyResponseException, ReqAbortException, RespParseException
from src.llm_models.payload_content.message import (
    ImageMessagePart,
    Message as InternalMessage,
    RoleType,
    TextMessagePart,
)
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption

# 延迟 import 以避免与 model_client/__init__.py 的循环依赖
if TYPE_CHECKING:
    from .model_client.base_client import APIResponse

UsageTuple = Tuple[int, ...]

logger = get_logger("llm_models")

RESPONSES_RESERVED_EXTRA_BODY_KEYS: set[str] = {
    "input",
    "max_output_tokens",
    "messages",
    "model",
    "response_format",
    "stream",
    "temperature",
    "text",
    "tools",
}
"""不应落入 ``extra_body`` 的字段集合（已由 SDK 原生参数承载）。"""


def _extract_text(message: InternalMessage) -> str:
    """提取消息中所有 TextMessagePart 并拼接。"""
    return "".join(part.text for part in message.parts if isinstance(part, TextMessagePart))


def convert_messages_to_response_input(messages: List[InternalMessage]) -> ResponseInputParam:
    """将内部消息列表转换为 Responses API 的 ``input`` 字段。"""
    from .model_client.openai_client import _normalize_image_part_for_openai

    converted: ResponseInputParam = []
    for message in messages:
        if message.role == RoleType.System:
            text = _extract_text(message)
            converted.append({"role": "system", "content": [{"type": "input_text", "text": text}] if text else []})

        elif message.role == RoleType.User:
            user_content: List[ResponseInputContentParam] = []
            for part in message.parts:
                if isinstance(part, TextMessagePart):
                    user_content.append({"type": "input_text", "text": part.text})
                else:
                    normalized_image = _normalize_image_part_for_openai(cast(ImageMessagePart, part))
                    if normalized_image is None:
                        user_content.append({"type": "input_text", "text": "[图片内容不可用]"})
                    else:
                        image_format, image_base64 = normalized_image
                        user_content.append({
                            "type": "input_image",
                            "image_url": f"data:image/{image_format};base64,{image_base64}",
                            "detail": "auto",
                        })
            converted.append({"role": "user", "content": user_content})

        elif message.role == RoleType.Assistant:
            assistant_text = _extract_text(message)
            if assistant_text:
                converted.append({"role": "assistant", "content": assistant_text})
            for tool_call in message.tool_calls or []:
                converted.append({
                    "type": "function_call",
                    "call_id": tool_call.call_id,
                    "name": tool_call.func_name,
                    "arguments": json.dumps(tool_call.args or {}, ensure_ascii=False),
                })

        elif message.role == RoleType.Tool:
            tool_text = _extract_text(message)
            converted.append({
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": tool_text,
            })

        else:
            raise ValueError(f"不支持的消息角色：{message.role}")

    return converted


def convert_tool_options(tool_options: List[ToolOption]) -> List[ToolParam]:
    """将工具定义转换为 Responses API 的扁平 function tool 列表。"""
    return [
        cast(ToolParam, {
            "type": "function",
            "name": opt.name,
            "description": opt.description,
            "parameters": cast(Dict[str, object], opt.parameters_schema or {"type": "object", "properties": {}}),
            "strict": False,
        })
        for opt in tool_options
    ]


def convert_response_format(response_format: RespFormat | None) -> ResponseTextConfigParam | None:
    """将内部响应格式转换为 Responses API 的 ``text`` 配置；TEXT 时返回 ``None``。"""
    if response_format is None or response_format.format_type == RespFormatType.TEXT:
        return None
    if response_format.format_type == RespFormatType.JSON_OBJ:
        return {"format": {"type": "json_object"}}
    if response_format.format_type == RespFormatType.JSON_SCHEMA:
        schema_wrapper = response_format.schema or {}
        raw_schema = schema_wrapper.get("schema")
        return {
            "format": {
                "type": "json_schema",
                "name": str(schema_wrapper.get("name") or "response_schema"),
                "schema": cast(Dict[str, object], raw_schema if isinstance(raw_schema, dict) else {}),
                "strict": bool(schema_wrapper.get("strict", False)),
            }
        }
    return None


def extract_usage_record(usage: Any) -> UsageTuple | None:
    """从 ``ResponseUsage`` 中提取统一 usage 五元组。"""
    if usage is None:
        return None
    prompt_tokens: int = usage.input_tokens
    cached_tokens: int = usage.input_tokens_details.cached_tokens
    miss_tokens = max(prompt_tokens - cached_tokens, 0) if cached_tokens else 0
    return (prompt_tokens, usage.output_tokens, usage.total_tokens, cached_tokens, miss_tokens)


def parse_response(
    resp: Response,
    *,
    reasoning_parse_mode: ReasoningParseMode,
    tool_argument_parse_mode: ToolArgumentParseMode,
) -> Tuple["APIResponse", UsageTuple | None]:
    """解析 Responses API 的非流式响应。"""
    from .model_client.base_client import APIResponse
    from .model_client.openai_client import _parse_tool_arguments

    output_items = resp.output
    if not output_items:
        raise EmptyResponseException(resp, "响应解析失败，output 为空或缺失")

    api_response = APIResponse()
    text_segments: List[str] = []
    reasoning_segments: List[str] = []
    tool_calls: List[ToolCall] = []

    for item in output_items:
        if item.type == "message":
            for content_part in item.content:
                if content_part.type == "output_text":
                    text_segments.append(content_part.text)
                elif content_part.type == "refusal" and content_part.refusal:
                    text_segments.append(content_part.refusal)

        elif item.type == "function_call":
            tool_calls.append(ToolCall(
                call_id=item.call_id,
                func_name=item.name,
                args=_parse_tool_arguments(item.arguments, tool_argument_parse_mode, resp),
            ))

        elif item.type == "reasoning" and reasoning_parse_mode != ReasoningParseMode.NONE:
            reasoning_segments.extend(cp.text for cp in (item.content or []))
            reasoning_segments.extend(sp.text for sp in item.summary)

    if reasoning_segments:
        api_response.reasoning_content = "".join(reasoning_segments).strip() or None
    if text_segments:
        api_response.content = "".join(text_segments).strip() or None
    if tool_calls:
        api_response.tool_calls = tool_calls
    api_response.raw_data = resp

    if resp.incomplete_details is not None and resp.incomplete_details.reason == "max_output_tokens":
        logger.info(
            f"模型{resp.model}因为超过最大 max_output_tokens 限制，可能仅输出部分内容，可视情况调整"
        )

    if not api_response.content and not api_response.tool_calls:
        raise EmptyResponseException(resp)

    return api_response, extract_usage_record(resp.usage)


@dataclass(slots=True)
class _StreamedFunctionCall:
    """流式 function_call 累积状态。"""

    call_id: str = ""
    name: str = ""
    arguments: List[str] = field(default_factory=list)


class _ResponsesStreamAccumulator:
    """Responses API 流式响应累积器。"""

    def __init__(
        self,
        *,
        reasoning_parse_mode: ReasoningParseMode,
        tool_argument_parse_mode: ToolArgumentParseMode,
    ) -> None:
        self.reasoning_parse_mode = reasoning_parse_mode
        self.tool_argument_parse_mode = tool_argument_parse_mode
        self.content_chunks: List[str] = []
        self.reasoning_chunks: List[str] = []
        self.function_calls: Dict[int, _StreamedFunctionCall] = {}
        self.usage_record: UsageTuple | None = None
        self.model_name: str | None = None
        self.incomplete_reason: str | None = None

    def _function_call_state(self, output_index: int) -> _StreamedFunctionCall:
        return self.function_calls.setdefault(output_index, _StreamedFunctionCall())

    def process_event(self, event: ResponseStreamEvent) -> None:
        event_type = event.type

        if event_type == "response.output_text.delta":
            if event.delta:
                self.content_chunks.append(event.delta)

        elif event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
            if self.reasoning_parse_mode != ReasoningParseMode.NONE and event.delta:
                self.reasoning_chunks.append(event.delta)

        elif event_type in {"response.output_item.added", "response.output_item.done"}:
            item = event.item
            if item.type == "function_call":
                state = self._function_call_state(event.output_index)
                if not state.call_id:
                    state.call_id = item.call_id
                if not state.name:
                    state.name = item.name
                if event_type == "response.output_item.done" and item.arguments and not state.arguments:
                    state.arguments.append(item.arguments)

        elif event_type == "response.function_call_arguments.delta":
            if event.delta:
                self._function_call_state(event.output_index).arguments.append(event.delta)

        elif event_type == "response.completed":
            response_obj = event.response
            self.model_name = response_obj.model or self.model_name
            self.usage_record = extract_usage_record(response_obj.usage)
            if response_obj.incomplete_details is not None:
                self.incomplete_reason = response_obj.incomplete_details.reason or self.incomplete_reason

        elif event_type == "response.failed":
            error = event.response.error
            raise RespParseException(
                event.response,
                error.message if error is not None else "Responses API 流式请求失败",
            )

    def build_response(self) -> "APIResponse":
        from .model_client.base_client import APIResponse
        from .model_client.openai_client import _parse_tool_arguments

        response = APIResponse()
        if self.reasoning_chunks:
            response.reasoning_content = "".join(self.reasoning_chunks).strip() or None
        if self.content_chunks:
            response.content = "".join(self.content_chunks).strip() or None

        if self.function_calls:
            response.tool_calls = []
            for output_index in sorted(self.function_calls):
                state = self.function_calls[output_index]
                if not state.name:
                    raise RespParseException(None, f"响应解析失败，工具调用 {output_index} 缺少函数名。")
                raw_arguments = "".join(state.arguments).strip()
                arguments = (
                    _parse_tool_arguments(raw_arguments, self.tool_argument_parse_mode, None)
                    if raw_arguments
                    else None
                )
                call_id = state.call_id or f"tool_call_{output_index}_{uuid4().hex}"
                response.tool_calls.append(
                    ToolCall(call_id=call_id, func_name=state.name, args=arguments)
                )

        response.raw_data = {"model": self.model_name} if self.model_name else None

        if self.incomplete_reason == "max_output_tokens":
            logger.info(
                f"模型{self.model_name or ''}因为超过最大 max_output_tokens 限制，可能仅输出部分内容，可视情况调整"
            )

        if not response.content and not response.tool_calls:
            raise EmptyResponseException(response.raw_data)
        return response


async def default_stream_response_handler(
    resp_stream: AsyncStream[ResponseStreamEvent],
    interrupt_flag: asyncio.Event | None,
    *,
    reasoning_parse_mode: ReasoningParseMode,
    tool_argument_parse_mode: ToolArgumentParseMode,
) -> Tuple["APIResponse", UsageTuple | None]:
    """处理 Responses API 流式响应，累积所有事件后返回统一响应对象与可选 usage。"""
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=reasoning_parse_mode,
        tool_argument_parse_mode=tool_argument_parse_mode,
    )
    async for event in resp_stream:
        if interrupt_flag and interrupt_flag.is_set():
            raise ReqAbortException("请求被外部信号中断")
        accumulator.process_event(event)
    return accumulator.build_response(), accumulator.usage_record
