# GmeQwen2VL 使用说明

使用 `GmeQwen2VL` 计算文本、图像及多模态内容的嵌入（embeddings），并进行相似性计算和信息检索任务。

## 1. 计算文本嵌入  
调用 `get_text_embeddings()` 计算文本的向量表示。  
```python
text_embeddings = gme.get_text_embeddings(texts=texts)
```

---

## 2. 计算图像嵌入  
调用 `get_image_embeddings()` 计算图像的向量表示。  
```python
image_embeddings = gme.get_image_embeddings(images=images)
```

---

## 3. 计算文本与图像的相似度  
使用点积计算文本和图像嵌入之间的相似性。  
```python
similarity = paddle.sum(text_embeddings * image_embeddings, axis=-1)
```

---

## 4. 计算带有自定义指令的文本嵌入  
可以使用自定义指令（例如 "Find an image that matches the given text."）计算文本嵌入：  
```python
e_query = gme.get_text_embeddings(
    texts=texts,
    instruction="Find an image that matches the given text."
)
```

对图像嵌入的计算需设置 `is_query=False`：  
```python
e_corpus = gme.get_image_embeddings(images=images, is_query=False)
similarity_query = paddle.sum(e_query * e_corpus, axis=-1)
```

---

## 5. 计算融合嵌入（文本+图像）  
计算文本和图像的融合表示，并计算其相似性：  
```python
fused_embeddings = gme.get_fused_embeddings(texts=texts, images=images)
similarity_fused_embeddings = paddle.sum(
    fused_embeddings[0] * fused_embeddings[1], axis=-1
)
```

---

## 6. 信息检索任务：查询文本嵌入  
模拟信息检索任务，计算查询文本的嵌入：  
```python
queries = ["Find an image of a Tesla Cybertruck."]
e_query = gme.encode_queries(queries)
```

---

## 7. 信息检索任务：数据库文本嵌入  
计算数据库中的文本嵌入（例如不同车型的描述信息）：  
```python
corpus = [
    {"title": "Tesla Cybertruck", "text": "A battery electric pickup truck by Tesla."},
    {"title": "Ford F-150", "text": "A popular American pickup truck."}
]
e_corpus = gme.encode_corpus(corpus)
```

---

## 8. 计算查询与数据库之间的相似度  
计算查询文本与数据库文本的匹配分数：  
```python
similarity = paddle.sum(e_query * e_corpus, axis=-1)
```