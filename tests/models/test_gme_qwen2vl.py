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

import unittest
import paddle
import numpy as np
from PIL import Image
from paddlemix.models.gme_qwen2_vl.modeling_gme_qwen2_vl import GmeQwen2VL


class TestGmeQwen2VL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the model before running tests to avoid redundant loading."""
        cls.model = GmeQwen2VL(model_name="GME-Qwen2-VL/gme-Qwen2-VL-2B-Instruct")
    
    def test_text_embedding(self):
        """Test text embedding computation."""
        texts = ["What kind of car is this?", "The Tesla Cybertruck is a battery electric pickup truck."]
        embeddings = self.model.get_text_embeddings(texts=texts)
        
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(texts))  # Ensure output matches input text count

    def test_image_embedding(self):
        """Test image embedding computation."""
        images = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]
        embeddings = self.model.get_image_embeddings(images=images)
        
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(images))  # Ensure output matches input image count

    def test_text_image_similarity(self):
        """Test similarity calculation between text and image embeddings."""
        texts = ["What kind of car is this?"]
        images = [Image.new("RGB", (224, 224))]
        
        text_embeddings = self.model.get_text_embeddings(texts=texts)
        image_embeddings = self.model.get_image_embeddings(images=images)
        
        similarity = paddle.sum(text_embeddings * image_embeddings, axis=-1)
        
        self.assertIsNotNone(similarity)
        self.assertEqual(similarity.shape, [len(texts)])  # One-to-one similarity, output should match text count

    def test_text_embedding_with_instruction(self):
        """Test text embedding computation with a custom instruction."""
        texts = ["Find an image of a Tesla Cybertruck."]
        instruction = "Find an image that matches the given text."
        
        embeddings = self.model.get_text_embeddings(texts=texts, instruction=instruction)
        
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(texts))

    def test_fused_embedding(self):
        """Test fused embedding computation for text and image."""
        texts = ["What kind of car is this?"]
        images = [Image.new("RGB", (224, 224))]
        
        fused_embeddings = self.model.get_fused_embeddings(texts=texts, images=images)

        self.assertIsNotNone(fused_embeddings)
        self.assertEqual(fused_embeddings.shape[-1], 1536)  # Ensure embedding size is correct

    def test_query_embedding(self):
        """Test query text embedding computation."""
        queries = ["Find an image of a Tesla Cybertruck."]
        query_embeddings = self.model.encode_queries(queries)
        
        self.assertIsNotNone(query_embeddings)
        self.assertEqual(query_embeddings.shape[0], len(queries))

    def test_corpus_embedding(self):
        """Test corpus text embedding computation."""
        corpus = [
            {"title": "Tesla Cybertruck", "text": "A battery electric pickup truck by Tesla."},
            {"title": "Ford F-150", "text": "A popular American pickup truck."}
        ]
        corpus_embeddings = self.model.encode_corpus(corpus)
        
        self.assertIsNotNone(corpus_embeddings)
        self.assertEqual(corpus_embeddings.shape[0], len(corpus))

    def test_query_corpus_similarity(self):
        """Test similarity calculation between query and corpus embeddings."""
        queries = ["Find an image of a Tesla Cybertruck."]
        corpus = [
            {"title": "Tesla Cybertruck", "text": "A battery electric pickup truck by Tesla."},
            {"title": "Ford F-150", "text": "A popular American pickup truck."}
        ]
        
        e_query = self.model.encode_queries(queries)  # Shape: [1, 1536]
        e_corpus = self.model.encode_corpus(corpus)  # Shape: [2, 1536]
        similarity = paddle.sum(e_query * e_corpus, axis=-1)  # Shape: [2]

        self.assertIsNotNone(similarity)
        self.assertEqual(similarity.shape[0], len(corpus))  # Fix: Expected shape should match the corpus size


if __name__ == "__main__":
    unittest.main()