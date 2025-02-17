import os
import paddle
"""
Image Reconstruction code
"""
import sys
sys.path.append(os.getcwd())
from omegaconf import OmegaConf
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from taming.models.lfqgan import VQModel
import argparse



def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(**config.model.init_args)
    if ckpt_path is not None:
        sd = paddle.load(path=str(ckpt_path))
        missing, unexpected = model.set_state_dict(state_dict=sd)
    model.eval()
    return model


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'class_path' in config:
        raise KeyError('Expected key `class_path` to instantiate.')
    return get_obj_from_str(config['class_path'])(**config.get('init_args',
        dict()))


def custom_to_pil(x):
    x = x.detach().cpu()
    x = paddle.clip(x=x, min=-1.0, max=1.0)
    x = (x + 1.0) / 2.0
    x = x.transpose(perm=[1, 2, 0]).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == 'RGB':
        x = x.convert('RGB')
    return x


def main(args):
    config_file = args.config_file
    configs = OmegaConf.load(config_file)
    configs.data.init_args.batch_size = args.batch_size
    configs.data.init_args.test.params.config.size = args.image_size
    configs.data.init_args.test.params.config.subset = args.subset
    model = load_vqgan_new(configs, args.ckpt_path)
    visualize_dir = args.save_dir
    visualize_version = args.version
    visualize_original = os.path.join(visualize_dir, visualize_version,
        'original_{}'.format(args.image_size))
    visualize_rec = os.path.join(visualize_dir, visualize_version, 'rec_{}'
        .format(args.image_size))
    if not os.path.exists(visualize_original):
        os.makedirs(visualize_original, exist_ok=True)
    if not os.path.exists(visualize_rec):
        os.makedirs(visualize_rec, exist_ok=True)
    configs.data['init_args'].pop("train")
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()
    count = 0
    with paddle.no_grad():
        for idx, batch in tqdm(enumerate(dataset._val_dataloader())):
            if count > args.image_num:
                break
            images = batch['image'].transpose(perm=[0, 3, 1, 2])
            count += tuple(images.shape)[0]
            if model.use_ema:
                with model.ema_scope():
                    reconstructed_images, _, _ = model(images)
            image = images[0]
            reconstructed_image = reconstructed_images[0]

            image = custom_to_pil(image)
            reconstructed_image = custom_to_pil(reconstructed_image)
            image.save(os.path.join(visualize_original, '{}.png'.format(idx)))
            reconstructed_image.save(os.path.join(visualize_rec, '{}.png'.
                format(idx)))


def get_args():
    parser = argparse.ArgumentParser(description='inference parameters')
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--ckpt_path', required=True, type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--image_num', default=50, type=int)
    parser.add_argument('--subset', default=None)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
