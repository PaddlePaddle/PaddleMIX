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


from enum import Enum
from typing import Dict, Any, TypeVar

from ..misc import JsonSchemaValidator


T = TypeVar('T', bound=Dict[str, Any])


class SCHEMA(Enum):
    # data-type
    MM = \
"""
$id: 'https://multi-modality-llm-schema'
$schema: 'https://json-schema.org/draft/2020-12/schema'

type: object
properties:
  id:
    anyOf:
      - 
        type: string
        pattern: '\S{1,}'
      - 
        type: integer
        minimum: 0
  
  image:
    type: string
    pattern: '\.(jpg|jpeg|png|JPG|JPEG|PNG)$'
    description: 'path ends with .png or .jpg or .jpeg'

  conversations:
    type: array
    minItems: 1
    items:
      type: object
      properties:
        from:
          type: string
          enum: ['human', 'gpt']
        value:
          type: string
          pattern: '\S{1,}'
          description: 'Non whitespace characters must be at least 1 in length'
      required: ['from', 'value']

  meta: 
    type: 'object'
    properties:
      width: 
        type: ['integer', 'null']
      height:
        type: ['integer', 'null']
      is_valid:
        type: 'boolean'
    required: ['width', 'height', 'is_valid']

required: ['id', 'image', 'conversations']
"""

    MIX = \
"""
$id: 'https://example.com/schemas/multimodal_mix'
$schema: 'https://json-schema.org/draft/2020-12/schema'

type: object
properties:
  id:
    anyOf:
      - 
        type: string
        pattern: '\S{1,}'
      - 
        type: integer
        minimum: 0

  images:
    anyOf:
      - 
        type: 'null'
      - 
        type: array
        minItems: 1
        items:
          type: object
          properties:
            id:
              type: integer
              minimum: 0
            url:
              type: string
              pattern: '\.(jpg|jpeg|png|webp|JPG|JPEG|PNG)$'
              description: '.png or .jpg or .jpeg or .webp'
            height:
              type: integer
              minimum: 0
            width:
              type: integer
              minimum: 0
          required:
            - id
            - url
            
  conversations:
    type: array
    minItems: 1
    items:
      type: object
      properties:
        from:
          type: string
          description: 'user or assistant'
          enum: 
            - user
            - assistant
        value:
          anyOf:
            - type: string
            - type: 'null'
      required:
        - from
        - value
required:
  - id
  - images
  - conversations
"""

SCHEMA_VALIDATORS = {
    k: JsonSchemaValidator.from_string(k.value) for k in SCHEMA
}

def is_valid_schema(item: T, schema: SCHEMA, suppress_exceptions: bool=False) -> bool:
    if suppress_exceptions:
        try:
            SCHEMA_VALIDATORS[schema].validate(item)
            return True
        except Exception as e:
            return False
    else:
        SCHEMA_VALIDATORS[schema].validate(item)
        return True

def assert_mm_schema(item) -> None:
    assert is_valid_schema(item, SCHEMA.MM), f'{item} is not valid mm schema'

