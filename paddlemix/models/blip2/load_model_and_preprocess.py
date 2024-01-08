import paddle
from blip2_opt2_instruct import Blip2OptInstruct
import logging
from paddlemix.processors import Blip2ImageTrainProcessor, BlipImageEvalProcessor, BlipCaptionProcessor
from paddlemix.processors.base_processor import BaseProcessor

from omegaconf import OmegaConf



def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        if cfg.name == 'blip2_image_train':
            return Blip2ImageTrainProcessor.from_config(cfg)
        elif cfg.name == 'blip_image_eval':
            return BlipImageEvalProcessor.from_config(cfg)
        elif cfg.name == 'blip_caption':
            return BlipCaptionProcessor.from_config(cfg)
        else:
            return BaseProcessor()

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors

def load_model_and_preprocess(model_type, is_eval=False, device="cpu"):
    model_cls = Blip2OptInstruct
    # load model
    model = model_cls.from_pretrained(pretrained_model_name_or_path=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load('/home/aistudio/work/PaddleMIX/paddlemix/models/blip2/blip2_instruct_opt.yaml')
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model  ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == paddle.device.get_device():
        model = model.float()

    return model.to(device), vis_processors, txt_processors

if __name__ == '__main__':
    load_model_and_preprocess(model_type='facebook/opt-2.7b')
