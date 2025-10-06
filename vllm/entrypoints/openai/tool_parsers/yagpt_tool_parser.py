# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.chat_utils import make_tool_call_id
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

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []  # track streamed args for each tool

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

        if self.bot_seq not in current_text:
            # Wait before full toll call prefix is generated
            if self.bot_seq.startswith(current_text):
                return None
            return DeltaMessage(content=delta_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            groups = current_text.split(self.bot_seq)[1:]

            for group in groups:
                if "[NL]" in group:
                    name, arguments_str = group.split("[NL]", 1)
                else:
                    return None

                if not arguments_str:
                    tool_call_arr.append({"name": name})
                    is_complete.append(False)

                try:
                    # Try to parse the arguments JSON (partial or complete)
                    args_obj, _ = partial_json_loads(arguments_str, flags)
                    tool_call_arr.append({
                        "name": name.strip(),
                        "arguments": args_obj
                    })
                    is_complete.append(is_complete_json(arguments_str))
                except partial_json_parser.core.exceptions.MalformedJSON:
                    # Arguments JSON is not parseable yet
                    tool_call_arr.append({"name": name})
                    is_complete.append(False)

            # select as the current tool call the one we're on the state at
            current_tool_call: dict = tool_call_arr[self.current_tool_id] \
                if len(tool_call_arr) > 0 else {}

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (len(tool_call_arr) > 0
                  and len(tool_call_arr) > self.current_tool_id + 1):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments,
                                                   ensure_ascii=False)
                        sent = len(
                            self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        logger.debug("got arguments diff: %s", argument_diff)
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=argument_diff).
                                          model_dump(exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:

                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=make_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(
                        self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments,
                                               ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[
                        self.current_tool_id].get("arguments")

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments,
                                                    ensure_ascii=False)
                        if cur_args_json != prev_args_json:

                            prefix = find_common_prefix(
                                prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=argument_diff).
                                          model_dump(exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error")
            return None
