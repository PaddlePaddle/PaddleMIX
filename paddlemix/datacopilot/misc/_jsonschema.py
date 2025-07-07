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


import importlib.metadata

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

version = importlib.metadata.version("jsonschema")


class JsonSchemaValidator(object):
    def __init__(self, schema):
        self._schema = schema
        self._validator = None

    @classmethod
    def from_string(cls, s: str) -> "JsonSchemaValidator":
        schema = yaml.safe_load(s)
        return cls(schema)

    @classmethod
    def from_yaml(cls, path: str) -> "JsonSchemaValidator":
        with open(path, "r") as f:
            schema = yaml.load(f, Loader=yaml.Loader)
        return cls(schema)

    def validate(self, instance):
        self.validator.validate(instance)

    @property
    def schema(self):
        return self._schema

    @property
    def validator(self) -> Draft202012Validator:
        if self._validator is not None:
            return self._validator

        # resource = DRAFT202012.create_resource(content)
        resource = Resource(contents=self.schema, specification=DRAFT202012)
        registry = Registry().with_resource(uri=self.schema["$id"], resource=resource)
        # self._validator = Draft202012Validator(self.schema, registry=registry,)
        self._validator = Draft202012Validator(
            {"$ref": self.schema["$id"]},
            registry=registry,
        )

        return self._validator
