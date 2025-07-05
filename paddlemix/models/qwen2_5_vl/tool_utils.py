# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from typing_extensions import override

SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

FunctionCall = namedtuple("FunctionCall", ["name", "arguments"])


@dataclass
class ToolUtils(ABC):
    """
    Base class for tool utilities.
    """

    # @staticmethod
    # @abstractmethod
    # def get_function_slots() -> SLOTS:
    #     r"""
    #     Gets a list of slots corresponding to a single function call.
    #     """
    #     ...

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        r"""
        Generates the system message describing all the available tools.
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        r"""
        Extracts all the function calls from the response message.
        """
        ...


class DefaultToolUtils(ToolUtils):
    @override
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["Action: {{name}}\nAction Input: {{arguments}}\n"]

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class QwenToolUtils(ToolUtils):
    r"""Qwen 2.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(
                "<tool_call>\n" + f'{{"name": "{name}", "arguments": {arguments}}}' + "\n</tool_call>"
            )

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: list[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


TOOLS = {"default": DefaultToolUtils(), "qwen": QwenToolUtils()}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError("Tool utils `{}` not found.".format(name))

    return tool_utils
