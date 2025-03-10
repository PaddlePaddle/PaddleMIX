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

import paddle
import paddle.nn.functional as F
from paddlemix.models.gme_qwen2_vl.modeling_gme_qwen2_vl import GmeQwen2VL


if __name__ == "__main__":
    """
    Example code: Using `GmeQwen2VL` to compute embeddings for text, images, and multimodal content
    """

    # **1. Initialize model**
    print("\n=== Initializing GmeQwen2VL model ===")
    gme = GmeQwen2VL(model_name="GME-Qwen2-VL/gme-Qwen2-VL-2B-Instruct")
    
    # **2. Define text and images**
    texts = [
        "What kind of car is this?",
        "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    ]
    images = [
        'https://en.wikipedia.org/wiki/File:Tesla_Cybertruck_damaged_window.jpg',
        'https://en.wikipedia.org/wiki/File:2024_Tesla_Cybertruck_Foundation_Series,_front_left_(Greenwich).jpg',
    ]

    # **3. Compute text embeddings**
    print("\n=== Computing text embeddings ===")
    text_embeddings = gme.get_text_embeddings(texts=texts)
    print(f"text_embeddings:\n{text_embeddings}")

    # **4. Compute image embeddings**
    print("\n=== Computing image embeddings ===")
    image_embeddings = gme.get_image_embeddings(images=images)
    print(f"image_embeddings:\n{image_embeddings}")

    # **5. Calculate similarity between text and images**
    print("\n=== Computing text-image similarity ===")
    similarity = paddle.sum(text_embeddings * image_embeddings, axis=-1)
    print(f"similarity:\n{similarity}")

    # **6. Compute text embeddings with custom instruction**
    print("\n=== Computing text embeddings with custom instruction ===")
    e_query = gme.get_text_embeddings(texts=texts, instruction="Find an image that matches the given text.")
    e_corpus = gme.get_image_embeddings(images=images, is_query=False)
    similarity_query = paddle.sum(e_query * e_corpus, axis=-1)
    print(f"similarity_query:\n{similarity_query}")

    # **7. Compute fused embeddings for text+image**
    print("\n=== Computing fused embeddings for text+image ===")
    fused_embeddings = gme.get_fused_embeddings(texts=texts, images=images)
    similarity_fused_embeddings = paddle.sum(fused_embeddings[0] * fused_embeddings[1], axis=-1)
    print(f"similarity_fused_embeddings:\n{similarity_fused_embeddings}")

    # **8. Simulate information retrieval task: text query**
    print("\n=== Simulating information retrieval task: text query ===")
    queries = ["Find an image of a Tesla Cybertruck."]
    e_query = gme.encode_queries(queries)
    print(f"Query embeddings:\n{e_query}")

    # **9. Simulate information retrieval task: database text**
    print("\n=== Simulating information retrieval task: database text ===")
    corpus = [
        {"title": "Tesla Cybertruck", "text": "A battery electric pickup truck by Tesla."},
        {"title": "Ford F-150", "text": "A popular American pickup truck."}
    ]
    e_corpus = gme.encode_corpus(corpus)
    print(f"Corpus embeddings:\n{e_corpus}")

    # **10. Calculate similarity between query and database**
    print("\n=== Calculate similarity between query and database ===")
    similarity = paddle.sum(e_query * e_corpus, axis=-1)
    print(f"Similarity scores:\n{similarity}")

    print("\n=== Computation completed! ===")