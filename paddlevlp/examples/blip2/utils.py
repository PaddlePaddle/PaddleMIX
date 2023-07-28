import os
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from paddlevlp.utils.downloader import get_weights_path_from_url
from paddlevlp.utils.downloader import is_url
from paddlevlp.models.blip2.eva_vit import interpolate_pos_embed
import paddle


class BlipCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlevlp.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        if "text_input" not in data_list[0].keys():
            text = None
        else:
            text = [sample["text_input"] for sample in data_list]
        image_id = [sample["image_id"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=32,
            return_tensors="pd",
            return_attention_mask=True,
            mode=self.mode, )
        batch.update({'image_id': image_id})
        return batch


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val":
        "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test":
        "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    #download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames['test'])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def load_pretrained_model(model, pretrained_model_path):
    if pretrained_model_path is None:
        return

    if not os.path.exists(pretrained_model_path):
        ValueError("Cannot find pretrained model path: {}".format(
            pretrained_model_path))

    if os.path.isfile(pretrained_model_path):
        path = pretrained_model_path
    elif is_url(pretrained_model_path):
        path = get_weights_path_from_url(pretrained_model_path)
    else:
        assert os.path.exists(
            pretrained_model_path), f"{pretrained_model_path} not exist"
    state_dict = paddle.load(path)
    interpolate_pos_embed(model, state_dict)
    model.set_state_dict(state_dict)
