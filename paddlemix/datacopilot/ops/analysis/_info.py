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


from rich.console import Console
from rich.table import Table

from ...core import MMDataset, register


@register(force=True)
def info(dataset: MMDataset) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    # table.add_column("Attr", style="dim", width=12)
    table.add_column("Attr", justify="left")
    table.add_column("Value", justify="left")

    table.add_row("Length", str(len(dataset)))

    console.print(dataset[0])
    console.print(table)


@register(force=True)
def head(dataset: MMDataset, n=10) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("id", style="dim", width=12)
    table.add_column("image", justify="right")
    table.add_column("conversations", justify="right")

    for i, item in enumerate(dataset[:n]):
        table.add_row(item["id"], item["image"], str(item["conversations"]))

    console.print(table)
