# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn.functional as F
from tqdm import tqdm

from paddlemix.processors.tokenizer import tokenize


def zero_shot_classifier(model, classnames_filename, templates_filename, args, text_tower=None):
    classnames = [i.strip() for i in open(classnames_filename).readlines()]
    templates = [i.strip() for i in open(templates_filename).readlines()]

    if text_tower is None:
        if hasattr(model, "_layers"):
            text_tower = model._layers.encode_text
        else:
            text_tower = model.encode_text
    tokenizer = tokenize
    with paddle.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = tokenizer(texts)  # tokenize
            class_embeddings = text_tower(texts)
            class_embedding = F.normalize(class_embeddings, axis=-1).mean(0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = paddle.stack(zeroshot_weights, axis=1)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    pred = output.topk(maxk, 1, True, True)[1].t().cast("int32")
    correct = pred == target.reshape([1, -1]).expand_as(pred)
    return [
        float(correct[: min(k, maxk)].reshape([-1]).astype(paddle.float32).sum(0, keepdim=True).numpy() * 100.0)
        for k in topk
    ]


class DummyAutocast:
    def __init__(self, *args, **kwargs):
        return

    def __enter__(self, *args, **kwargs):
        return

    def __exit__(self, *args, **kwargs):
        return


def get_autocast(precision):
    if precision == "float16":
        return paddle.amp.auto_cast
    elif precision == "bfloat16":
        return lambda: paddle.amp.auto_cast(dtype="bfloat16")
    else:
        return DummyAutocast


def get_cast_dtype(args):
    cast_dtype = None
    if args.bf16:
        cast_dtype = "bfloat16"
    elif args.fp16:
        cast_dtype = "float16"
    return cast_dtype


class ClipZeroShot:
    def __init__(self, model, args):
        data_path = args.classification_eval.strip()
        classname_filename = f"{data_path}/labels.txt"
        template_filename = f"{data_path}/templates.txt"

        self.data_name = os.path.basename(args.classification_eval)
        classifier_filename = (
            f"{os.path.dirname(classname_filename)}/{args.pretrained_text_model}_{self.data_name}_classifier.pdparams"
        )
        if os.path.exists(classifier_filename):
            print("load classifier from disk")
            classifier = paddle.load(classifier_filename)
        else:
            print("constructing classifier: {}.".format(classifier_filename))
            classifier = zero_shot_classifier(model, classname_filename, template_filename, args)
            paddle.save(classifier, classifier_filename)
        print(f"zero-shot evaluating classification task: {self.data_name}")
        if args.bf16:
            self.classifier = classifier.astype(paddle.bfloat16)
        elif args.fp16:
            self.classifier = classifier.astype(paddle.float16)
        else:
            self.classifier = classifier
        self.batch_size = args.per_device_eval_batch_size
        self.cast_dtype = get_cast_dtype(args)

    def zero_shot_eval(self, evalres):
        results = {}
        print("Extract features done, starting zero-shot classification evaluation.")
        predictions, labels = evalres.predictions, evalres.label_ids
        n = predictions.shape[0]
        top1, top5 = 0.0, 0.0

        autocast = get_autocast(self.cast_dtype)
        with paddle.no_grad():
            for step in tqdm(range((predictions.shape[0] + self.batch_size - 1) // self.batch_size)):
                with autocast():
                    image_features = paddle.to_tensor(
                        predictions[step * self.batch_size : (step + 1) * self.batch_size]
                    )
                    target = paddle.to_tensor(labels[step * self.batch_size : (step + 1) * self.batch_size])
                    logits = 100.0 * image_features @ self.classifier
                if logits.shape[-1] < 5:
                    (acc1,) = accuracy(logits, target, topk=(1,))
                    acc5 = -1
                else:
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
        top1 = top1 / n
        top5 = top5 / n
        results["val/imagenet-zeroshot-val-top1"] = top1
        results["val/imagenet-zeroshot-val-top5"] = top5

        results["top1"] = top1
        print(f"zero-shot classification task: {self.data_name}: top1: {top1}, top5: {top5}")
        print("Finished zero-shot evaluation.")

        return results
