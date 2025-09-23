# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (find_common_prefix,
                                                        is_complete_json,
                                                        partial_json_loads)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ToolParserManager.register_module("yagpt_json")
class YaGPTJsonToolParser(ToolParser):
    """
    Tool call parser for YaGPT models.

    Used when --enable-auto-tool-choice --tool-call-parser yagpt_json are set.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.bot_seq = "[NL][TOOL_CALL_START]"
        # Updated regex to match multiple JSONs separated by semicolons
        # This pattern is more robust and can handle nested JSON objects
        self.tool_call_regex = re.compile(
            r'{[^{}]*(?:{[^{}]*}[^{}]*)*}(?:\s*;\s*{[^{}]*(?:{[^{}]*}[^{}]*)*})*',
            re.DOTALL)

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        Only extracts JSON content and ignores any surrounding plain text.
        Supports both single JSON and multiple JSONs separated by semicolons.
        """
        # Quick check before running regex
        if not self.bot_seq in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        # Find JSON object(s) in the text using regex
        groups = model_output.split(self.bot_seq)[1:]
        try:
            tool_calls: list[ToolCall] = []
            for group in groups:
                name, arguments_str = group.split("[NL]", 1)
                _ = json.loads(arguments_str)  # check for a valid JSON
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=arguments_str
                        )
                    )
                )

            return ExtractedToolCallInformation(tools_called=True,
                                                tool_calls=tool_calls,
                                                content=None)

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        raise NotImplementedError(
            "Currently tool call parsing is supported only in non-streaming mode"
        )
