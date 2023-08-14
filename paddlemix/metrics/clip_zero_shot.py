import os
import paddle
import paddle.nn.functional as F
from tqdm import tqdm

from paddlemix.processors.tokenizer import tokenize


def zero_shot_classifier(model,
                         classnames_filename,
                         templates_filename,
                         args,
                         text_tower=None):
    classnames = [i.strip() for i in open(classnames_filename).readlines()]
    templates = [i.strip() for i in open(templates_filename).readlines()]

    if text_tower is None:
        if hasattr(model, '_layers'):
            text_tower = model._layers.module.encode_text if not hasattr(
                model._layers, 'encode_text') else model._layers.encode_text
        else:
            text_tower = model.module.encode_text if not hasattr(
                model, 'encode_text') else model.encode_text
    tokenizer = tokenize
    with paddle.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = tokenizer(texts)  # tokenize

            class_embeddings = text_tower(texts)
            class_embedding = F.normalize(class_embeddings, axis=-1).mean(0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = paddle.stack(zeroshot_weights, axis=1)
    return zeroshot_weights


def accuracy(output, target, topk=(1, )):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))
    return [
        float(correct[:k].reshape([-1]).astype(paddle.float32).sum(
            0, keepdim=True).numpy()) for k in topk
    ]


class DummyAutocast:
    def __init__(self, *args, **kwargs):
        return

    def __enter__(self, *args, **kwargs):
        return

    def __exit__(self, *args, **kwargs):
        return


def get_autocast(precision):
    if precision == 'float16':
        return paddle.amp.auto_cast
    elif precision == 'bfloat16':
        return lambda: paddle.amp.auto_cast(dtype='bfloat16')
    else:
        return DummyAutocast


def get_cast_dtype(args):
    cast_dtype = None
    if args.bf16:
        cast_dtype = 'bfloat16'
    elif args.fp16:
        cast_dtype = 'float16'
    return cast_dtype


def run(model, classifier, dataloader, args):
    cast_dtype = get_cast_dtype(args)
    autocast = get_autocast(cast_dtype)
    with paddle.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(
                dataloader, unit_scale=args.per_device_eval_batch_size):
            if cast_dtype is not None:
                images = images.cast(cast_dtype)
            target = target

            with autocast():
                if hasattr(model, '_layers'):
                    image_features = model._layers.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, axis=-1)
                logits = 100. * image_features @classifier

            # measure accuracy
            if logits.shape[-1] < 5:
                acc1, = accuracy(logits, target, topk=(1, ))
                acc5 = -1
            else:
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.shape[0]

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, args):
    results = {}
    print('Starting zero-shot classification evaluation.')
    print(f'Starting data: {data.keys()}.')
    for k, v in data.items():
        if 'eval/classification' in k:
            data_name = os.path.basename(k)
            classifier_filename = f'{os.path.dirname(v.classname_filename)}/{args.pretrained_text_model}_{data_name}_classifier.pt'
            if os.path.exists(classifier_filename):
                print('load classifier from disk')
                classifier = paddle.load(classifier_filename)
            else:
                print('constructing classifier.')
                classifier = zero_shot_classifier(model, v.classname_filename,
                                                  v.template_filename, args)
                paddle.save(classifier, classifier_filename)
            print(f"zero-shot evaluating classification task: {data_name}")
            if args.bf16:
                classifier = classifier.astype(paddle.bfloat16)
            elif args.fp16:
                classifier = classifier.astype(paddle.float16)

            top1, top5 = run(model, classifier, v.dataloader, args)
            results['val/imagenet-zeroshot-val-top1'] = top1
            results['val/imagenet-zeroshot-val-top5'] = top5

            #FIXME: DEBUG ONLY
            results[f'{k}-top1'] = top1
            print(
                f"zero-shot classification task: {data_name}: top1: {top1}, top5: {top5}"
            )

    print('Finished zero-shot evaluation.')

    return results
