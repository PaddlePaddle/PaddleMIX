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
import copy
import os
import tempfile
import time
import unittest

import numpy as np
import paddle
from huggingface_hub.repocard import RepoCard
from paddlenlp.transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
)
from ppdiffusers.loaders import (
    PADDLE_LORA_WEIGHT_NAME,
    TORCH_LORA_WEIGHT_NAME,
    AttnProcsLayers,
    LoraLoaderMixin,
    PatchedLoraProjection,
    text_encoder_attn_modules,
)
from ppdiffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_5,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
    XFormersAttnProcessor,
)
from ppdiffusers.utils import floats_tensor, is_ppxformers_available, str2bool
from ppdiffusers.utils.testing_utils import require_paddle_gpu, slow

lora_weights_name = TORCH_LORA_WEIGHT_NAME if str2bool(os.getenv("TO_DIFFUSERS", False)) else PADDLE_LORA_WEIGHT_NAME


def create_lora_layers(model, mock_weights: bool = True):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        if mock_weights:
            with paddle.no_grad():
                lora_attn_procs[name].to_q_lora.up.weight.set_value(lora_attn_procs[name].to_q_lora.up.weight + 1)
                lora_attn_procs[name].to_k_lora.up.weight.set_value(lora_attn_procs[name].to_k_lora.up.weight + 1)
                lora_attn_procs[name].to_v_lora.up.weight.set_value(lora_attn_procs[name].to_v_lora.up.weight + 1)
                lora_attn_procs[name].to_out_lora.up.weight.set_value(lora_attn_procs[name].to_out_lora.up.weight + 1)
    return lora_attn_procs


def create_unet_lora_layers(unet: paddle.nn.Layer):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_processor_class = (
            LoRAAttnProcessor2_5
            if hasattr(paddle.nn.functional, "scaled_dot_product_attention_")
            else LoRAAttnProcessor
        )
        lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    unet_lora_layers = AttnProcsLayers(lora_attn_procs)
    return lora_attn_procs, unet_lora_layers


def create_text_encoder_lora_attn_procs(text_encoder: paddle.nn.Layer):
    text_lora_attn_procs = {}
    lora_attn_processor_class = (
        LoRAAttnProcessor2_5 if hasattr(paddle.nn.functional, "scaled_dot_product_attention_") else LoRAAttnProcessor
    )
    for name, module in text_encoder_attn_modules(text_encoder):
        if isinstance(module.out_proj, paddle.nn.Linear):
            out_features = module.out_proj.weight.shape[1]
        elif isinstance(module.out_proj, PatchedLoraProjection):
            out_features = module.out_proj.regular_linear_layer.weight.shape[1]
        else:
            assert False, module.out_proj.__class__
        text_lora_attn_procs[name] = lora_attn_processor_class(hidden_size=out_features, cross_attention_dim=None)
    return text_lora_attn_procs


def create_text_encoder_lora_layers(text_encoder: paddle.nn.Layer):
    text_lora_attn_procs = create_text_encoder_lora_attn_procs(text_encoder)
    text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
    return text_encoder_lora_layers


def create_lora_3d_layers(model, mock_weights: bool = True):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        has_cross_attention = name.endswith("attn2.processor") and not (
            name.startswith("transformer_in") or "temp_attentions" in name.split(".")
        )
        cross_attention_dim = model.config.cross_attention_dim if has_cross_attention else None
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        elif name.startswith("transformer_in"):
            # Note that the `8 * ...` comes from: https://github.com/huggingface/diffusers/blob/7139f0e874f10b2463caa8cbd585762a309d12d6/src/diffusers/models/unet_3d_condition.py#L148
            hidden_size = 8 * model.config.attention_head_dim

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        if mock_weights:
            # add 1 to weights to mock trained weights
            with paddle.no_grad():
                lora_attn_procs[name].to_q_lora.up.weight.set_value(lora_attn_procs[name].to_q_lora.up.weight + 1)
                lora_attn_procs[name].to_k_lora.up.weight.set_value(lora_attn_procs[name].to_k_lora.up.weight + 1)
                lora_attn_procs[name].to_v_lora.up.weight.set_value(lora_attn_procs[name].to_v_lora.up.weight + 1)
                lora_attn_procs[name].to_out_lora.up.weight.set_value(lora_attn_procs[name].to_out_lora.up.weight + 1)
    return lora_attn_procs


def set_lora_weights(lora_attn_parameters, randn_weight=False, var=1.0):
    with paddle.no_grad():
        for parameter in lora_attn_parameters:
            if randn_weight:
                parameter[:] = paddle.randn(shape=parameter.shape, dtype=parameter.dtype) * var
            else:
                parameter.zero_()


def state_dicts_almost_equal(sd1, sd2):
    sd1 = dict(sorted(sd1.items()))
    sd2 = dict(sorted(sd2.items()))

    models_are_equal = True
    for ten1, ten2 in zip(sd1.values(), sd2.values()):
        if (ten1 - ten2).abs().max() > 1e-3:
            models_are_equal = False

    return models_are_equal


class LoraLoaderMixinTests(unittest.TestCase):
    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)
        text_encoder_lora_layers = create_text_encoder_lora_layers(text_encoder)

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        lora_components = {
            "unet_lora_layers": unet_lora_layers,
            "text_encoder_lora_layers": text_encoder_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = 32, 32

        generator = paddle.Generator().manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = paddle.randint(low=1, high=sequence_length, shape=(batch_size, sequence_length))

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def get_dummy_tokens(self):
        max_seq_length = 77
        inputs = paddle.randint(low=2, high=56, shape=(1, max_seq_length))
        prepared_inputs = {}
        prepared_inputs["input_ids"] = inputs
        return prepared_inputs

    def create_lora_weight_file(self, tmpdirname):
        _, lora_components = self.get_dummy_components()
        LoraLoaderMixin.save_lora_weights(
            save_directory=tmpdirname,
            unet_lora_layers=lora_components["unet_lora_layers"],
            text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
        )
        self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))

    def test_lora_save_load(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)
        _, _, pipeline_inputs = self.get_dummy_inputs()
        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(
            paddle.allclose(
                x=paddle.to_tensor(data=orig_image_slice), y=paddle.to_tensor(data=lora_image_slice)
            ).item()
        )

    # def test_lora_save_load_safetensors(self):
    #     pipeline_components, lora_components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionPipeline(**pipeline_components)
    #     sd_pipe.set_progress_bar_config(disable=None)
    #     _, _, pipeline_inputs = self.get_dummy_inputs()
    #     original_images = sd_pipe(**pipeline_inputs).images
    #     orig_image_slice = original_images[0, -3:, -3:, -1]
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         LoraLoaderMixin.save_lora_weights(
    #             save_directory=tmpdirname,
    #             unet_lora_layers=lora_components["unet_lora_layers"],
    #             text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
    #             safe_serialization=True,
    #         )
    #         self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
    #         sd_pipe.load_lora_weights(tmpdirname)
    #     lora_images = sd_pipe(**pipeline_inputs).images
    #     lora_image_slice = lora_images[0, -3:, -3:, -1]
    #     # Outputs shouldn't match.
    #     self.assertFalse(
    #         paddle.allclose(
    #             x=paddle.to_tensor(data=orig_image_slice), y=paddle.to_tensor(data=lora_image_slice)
    #         ).item()
    #     )

    def test_lora_save_load_legacy(self):
        pipeline_components, lora_components = self.get_dummy_components()
        unet_lora_attn_procs = lora_components["unet_lora_attn_procs"]
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)
        _, _, pipeline_inputs = self.get_dummy_inputs()
        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            unet = sd_pipe.unet
            unet.set_attn_processor(unet_lora_attn_procs)
            unet.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(
            paddle.allclose(
                x=paddle.to_tensor(data=orig_image_slice), y=paddle.to_tensor(data=lora_image_slice)
            ).item()
        )

    def test_text_encoder_lora_monkey_patch(self):
        pipeline_components, _ = self.get_dummy_components()
        pipe = StableDiffusionPipeline(**pipeline_components)
        dummy_tokens = self.get_dummy_tokens()
        # inference without lora
        outputs_without_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora.shape == [1, 77, 32]
        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)
        set_lora_weights(params, randn_weight=False)
        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == [1, 77, 32]
        assert paddle.allclose(
            x=outputs_without_lora, y=outputs_with_lora
        ).item(), "lora_up_weight are all zero, so the lora outputs should be the same to without lora outputs"
        # create lora_attn_procs with randn up.weights
        create_text_encoder_lora_attn_procs(pipe.text_encoder)
        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)
        set_lora_weights(params, randn_weight=True)
        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == [1, 77, 32]
        assert not paddle.allclose(
            x=outputs_without_lora, y=outputs_with_lora
        ).item(), "lora_up_weight are not zero, so the lora outputs should be different to without lora outputs"

    def test_text_encoder_lora_remove_monkey_patch(self):
        pipeline_components, _ = self.get_dummy_components()
        pipe = StableDiffusionPipeline(**pipeline_components)
        dummy_tokens = self.get_dummy_tokens()
        # inference without lora
        outputs_without_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora.shape == [1, 77, 32]
        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)
        set_lora_weights(params, randn_weight=True)
        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == [1, 77, 32]
        assert not paddle.allclose(
            x=outputs_without_lora, y=outputs_with_lora
        ).item(), "lora outputs should be different to without lora outputs"
        # remove monkey patch
        pipe._remove_text_encoder_monkey_patch()
        # inference with removed lora
        outputs_without_lora_removed = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora_removed.shape == [1, 77, 32]
        assert paddle.allclose(
            x=outputs_without_lora, y=outputs_without_lora_removed
        ).item(), "remove lora monkey patch should restore the original outputs"

    def test_text_encoder_lora_scale(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)
        _, _, pipeline_inputs = self.get_dummy_inputs()
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]
        lora_images_with_scale = sd_pipe(**pipeline_inputs, cross_attention_kwargs={"scale": 0.5}).images
        lora_image_with_scale_slice = lora_images_with_scale[0, -3:, -3:, -1]
        # Outputs shouldn't match.
        self.assertFalse(
            paddle.allclose(
                x=paddle.to_tensor(data=lora_image_slice), y=paddle.to_tensor(data=lora_image_with_scale_slice)
            ).item()
        )

    def test_lora_unet_attn_processors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.create_lora_weight_file(tmpdirname)
            pipeline_components, _ = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**pipeline_components)
            sd_pipe.set_progress_bar_config(disable=None)
            # check if vanilla attention processors are used
            for _, module in sd_pipe.unet.named_sublayers():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, (AttnProcessor, AttnProcessor2_5))
            # load LoRA weight file
            sd_pipe.load_lora_weights(tmpdirname)
            # check if lora attention processors are used
            for _, module in sd_pipe.unet.named_sublayers():
                # if isinstance(module, Attention):
                #     attn_proc_class = (
                #         LoRAAttnProcessor2_5
                #         if hasattr(paddle.nn.functional, "scaled_dot_product_attention_")
                #         else LoRAAttnProcessor
                #     )
                #     self.assertIsInstance(module.processor, attn_proc_class)
                if isinstance(module, Attention):
                    self.assertIsNotNone(module.to_q.lora_layer)
                    self.assertIsNotNone(module.to_k.lora_layer)
                    self.assertIsNotNone(module.to_v.lora_layer)
                    self.assertIsNotNone(module.to_out[0].lora_layer)

    def test_unload_lora_sd(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_lora_layers"].parameters(), randn_weight=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]
        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        original_images_two = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice_two = original_images_two[0, -3:, -3:, -1]
        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=0.001
        ), "Unloading LoRA parameters should lead to results similar to what was obtained with the pipeline without any LoRA parameters."

    def test_lora_unet_attn_processors_with_xformers(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.create_lora_weight_file(tmpdirname)
            pipeline_components, _ = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**pipeline_components)
            sd_pipe.set_progress_bar_config(disable=None)
            # enable XFormers
            sd_pipe.enable_xformers_memory_efficient_attention()
            # check if xFormers attention processors are used
            for _, module in sd_pipe.unet.named_sublayers():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, XFormersAttnProcessor)
            # load LoRA weight file
            sd_pipe.load_lora_weights(tmpdirname)
            # check if lora attention processors are used
            for _, module in sd_pipe.unet.named_sublayers():
                # if isinstance(module, Attention):
                #     self.assertIsInstance(module.processor, LoRAXFormersAttnProcessor)
                if isinstance(module, Attention):
                    self.assertIsNotNone(module.to_q.lora_layer)
                    self.assertIsNotNone(module.to_k.lora_layer)
                    self.assertIsNotNone(module.to_v.lora_layer)
                    self.assertIsNotNone(module.to_out[0].lora_layer)
            # unload lora weights
            sd_pipe.unload_lora_weights()
            # check if attention processors are reverted back to xFormers
            for _, module in sd_pipe.unet.named_sublayers():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, XFormersAttnProcessor)

    def test_lora_save_load_with_xformers(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)
        _, _, pipeline_inputs = self.get_dummy_inputs()
        # enable XFormers
        sd_pipe.enable_xformers_memory_efficient_attention()
        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]
        # Outputs shouldn't match.
        self.assertFalse(
            paddle.allclose(
                x=paddle.to_tensor(data=orig_image_slice), y=paddle.to_tensor(data=lora_image_slice)
            ).item()
        )


class SDXLLoraLoaderMixinTests(unittest.TestCase):
    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)
        text_encoder_one_lora_layers = create_text_encoder_lora_layers(text_encoder)
        text_encoder_two_lora_layers = create_text_encoder_lora_layers(text_encoder_2)

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
        }
        lora_components = {
            "unet_lora_layers": unet_lora_layers,
            "text_encoder_one_lora_layers": text_encoder_one_lora_layers,
            "text_encoder_two_lora_layers": text_encoder_two_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = paddle.Generator().manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = paddle.randint(low=1, high=sequence_length, shape=(batch_size, sequence_length))
        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_lora_save_load(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)
        _, _, pipeline_inputs = self.get_dummy_inputs()
        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]
        # Outputs shouldn't match.
        self.assertFalse(
            paddle.allclose(
                x=paddle.to_tensor(data=orig_image_slice), y=paddle.to_tensor(data=lora_image_slice)
            ).item()
        )

    def test_unload_lora_sdxl(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]
        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(tmpdirname)
        lora_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]
        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        original_images_two = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice_two = original_images_two[0, -3:, -3:, -1]
        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=0.001
        ), "Unloading LoRA parameters should lead to results similar to what was obtained with the pipeline without any LoRA parameters."

    def test_load_lora_locally(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=False,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.unload_lora_weights()

    # There may be only one item in one layer that needs to replace the name with an empty string.
    def replace_regular(self, keys):
        keys_new = []
        for i in keys:
            keys_new.append(i.replace("regular_linear_layer.", ""))
        return keys_new

    def test_text_encoder_lora_state_dict_unchanged(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)

        text_encoder_1_sd_keys = sorted(sd_pipe.text_encoder.state_dict().keys())
        text_encoder_2_sd_keys = sorted(sd_pipe.text_encoder_2.state_dict().keys())

        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=False,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

            text_encoder_1_sd_keys_2 = sorted(self.replace_regular(sd_pipe.text_encoder.state_dict().keys()))
            text_encoder_2_sd_keys_2 = sorted(self.replace_regular(sd_pipe.text_encoder_2.state_dict().keys()))

        sd_pipe.unload_lora_weights()

        text_encoder_1_sd_keys_3 = sorted(self.replace_regular(sd_pipe.text_encoder.state_dict().keys()))
        text_encoder_2_sd_keys_3 = sorted(self.replace_regular(sd_pipe.text_encoder_2.state_dict().keys()))

        # default & unloaded LoRA weights should have identical state_dicts
        assert text_encoder_1_sd_keys == text_encoder_1_sd_keys_3
        # default & loaded LoRA weights should NOT have identical state_dicts
        assert text_encoder_1_sd_keys != text_encoder_1_sd_keys_2

        # default & unloaded LoRA weights should have identical state_dicts
        assert text_encoder_2_sd_keys == text_encoder_2_sd_keys_3
        # default & loaded LoRA weights should NOT have identical state_dicts
        assert text_encoder_2_sd_keys != text_encoder_2_sd_keys_2

    def test_load_lora_locally_safetensors(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.unload_lora_weights()

    def test_lora_fusion(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        self.assertFalse(np.allclose(orig_image_slice, lora_image_slice, atol=1e-3))

    def test_unfuse_lora(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Reverse LoRA fusion.
        sd_pipe.unfuse_lora()
        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        orig_image_slice_two = original_images[0, -3:, -3:, -1]

        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "Fusion of LoRAs should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "Fusion of LoRAs should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=5e-3
        ), "Reversing LoRA fusion should lead to results similar to what was obtained with the pipeline without any LoRA parameters."

    def test_lora_fusion_is_not_affected_by_unloading(self):
        paddle.seed(0)
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        _ = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        images_with_unloaded_lora = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        images_with_unloaded_lora_slice = images_with_unloaded_lora[0, -3:, -3:, -1]

        assert (
            np.abs(lora_image_slice - images_with_unloaded_lora_slice).max() < 2e-1  # diffusers 0.23
        ), "`unload_lora_weights()` should have not effect on the semantics of the results as the LoRA parameters were fused."

    def test_fuse_lora_with_different_scales(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        _ = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.fuse_lora(lora_scale=1.0)
        lora_images_scale_one = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice_scale_one = lora_images_scale_one[0, -3:, -3:, -1]

        # Reverse LoRA fusion.
        sd_pipe.unfuse_lora()

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        sd_pipe.fuse_lora(lora_scale=0.5)
        lora_images_scale_0_5 = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        assert not np.allclose(
            lora_image_slice_scale_one, lora_image_slice_scale_0_5, atol=1e-03
        ), "Different LoRA scales should influence the outputs accordingly."

    def test_with_different_scales(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        original_images = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        original_imagee_slice = original_images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        lora_images_scale_one = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice_scale_one = lora_images_scale_one[0, -3:, -3:, -1]

        lora_images_scale_0_5 = sd_pipe(
            **pipeline_inputs, generator=paddle.Generator().manual_seed(0), cross_attention_kwargs={"scale": 0.5}
        ).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        lora_images_scale_0_0 = sd_pipe(
            **pipeline_inputs, generator=paddle.Generator().manual_seed(0), cross_attention_kwargs={"scale": 0.0}
        ).images
        lora_image_slice_scale_0_0 = lora_images_scale_0_0[0, -3:, -3:, -1]

        assert not np.allclose(
            lora_image_slice_scale_one, lora_image_slice_scale_0_5, atol=1e-03
        ), "Different LoRA scales should influence the outputs accordingly."

        assert np.allclose(
            original_imagee_slice, lora_image_slice_scale_0_0, atol=1e-03
        ), "LoRA scale of 0.0 shouldn't be different from the results without LoRA."

    def test_with_different_scales_fusion_equivalence(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        # sd_pipe.unet.set_default_attn_processor()
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        images = sd_pipe(
            **pipeline_inputs,
            generator=paddle.Generator().manual_seed(0),
        ).images
        images_slice = images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True, var=0.1)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True, var=0.1)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True, var=0.1)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, lora_weights_name))

        lora_images_scale_0_5 = sd_pipe(
            **pipeline_inputs,
            generator=paddle.Generator().manual_seed(0),
            cross_attention_kwargs={"scale": 0.5},
        ).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        sd_pipe.fuse_lora(lora_scale=0.5)
        lora_images_scale_0_5_fusion = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        lora_image_slice_scale_0_5_fusion = lora_images_scale_0_5_fusion[0, -3:, -3:, -1]

        assert np.allclose(
            lora_image_slice_scale_0_5, lora_image_slice_scale_0_5_fusion, atol=5e-03
        ), "Fusion shouldn't affect the results when calling the pipeline with a non-default LoRA scale."

        sd_pipe.unfuse_lora()
        images_unfused = sd_pipe(**pipeline_inputs, generator=paddle.Generator().manual_seed(0)).images
        images_slice_unfused = images_unfused[0, -3:, -3:, -1]

        assert np.allclose(images_slice, images_slice_unfused, atol=5e-03), "Unfused should match no LoRA"

        assert not np.allclose(
            images_slice, lora_image_slice_scale_0_5, atol=5e-03
        ), "0.5 scale and no scale shouldn't match"


# @deprecate_after_peft_backend
class UNet2DConditionLoRAModelTests(unittest.TestCase):
    model_class = UNet2DConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])
        encoder_hidden_states = floats_tensor((batch_size, 4, 32))

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_lora_processors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            sample1 = model(**inputs_dict).sample
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        model.set_attn_processor(model.attn_processors)
        with paddle.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        assert (sample1 - sample2).abs().max() < 3e-3
        assert (sample3 - sample4).abs().max() < 3e-3
        assert (sample2 - sample3).abs().max() > 3e-3

    def test_lora_save_load(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, to_diffusers=False)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            paddle.seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, from_diffusers=False)

        with paddle.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample - new_sample).abs().max() < 1e-4

        # LoRA and no LoRA should NOT be the same
        assert (sample - old_sample).abs().max() > 1e-4

    def test_lora_save_load_safetensors(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = (8, 16)

        paddle.seed(0)
        model = self.model_class(**init_dict)

        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, safe_serialization=True, to_diffusers=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            paddle.seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, from_diffusers=True, use_safetensors=True)
        with paddle.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
        assert (sample - new_sample).abs().max() < 0.0001
        assert (sample - old_sample).abs().max() > 0.0001

    # def test_lora_save_safetensors_load_torch(self):
    #     # enable deterministic behavior for gradient checkpointing
    #     init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

    #     init_dict["attention_head_dim"] = (8, 16)

    #     paddle.seed(0)
    #     model = self.model_class(**init_dict)

    #     lora_attn_procs = create_lora_layers(model, mock_weights=False)
    #     model.set_attn_processor(lora_attn_procs)
    #     # Saving as paddle, properly reloads with directly filename
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         model.save_attn_procs(tmpdirname, to_diffusers=True)
    #         self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
    #         paddle.seed(0)
    #         new_model = self.model_class(**init_dict)
    #         new_model.load_attn_procs(
    #             tmpdirname, weight_name="pytorch_lora_weights.bin", from_diffusers=True, use_safetensors=False
    #         )

    def test_lora_save_torch_force_load_safetensors_error(self):
        pass

    def test_lora_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
        model.set_default_attn_processor()
        with paddle.no_grad():
            new_sample = model(**inputs_dict).sample
        assert (sample - new_sample).abs().max() < 0.0001
        assert (sample - old_sample).abs().max() < 3e-3

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="scaled_dot_product_attention attention is only available with CUDA and `scaled_dot_product_attention` installed",
    )
    def test_lora_xformers_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["attention_head_dim"] = 8, 16
        paddle.seed(0)
        model = self.model_class(**init_dict)
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)
        with paddle.no_grad():
            sample = model(**inputs_dict).sample
            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample
            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample
        assert (sample - on_sample).abs().max() < 0.05
        assert (sample - off_sample).abs().max() < 0.05


# @deprecate_after_peft_backend
class UNet3DConditionModelTests(unittest.TestCase):
    model_class = UNet3DConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels, num_frames) + sizes)
        time_step = paddle.to_tensor([10])
        encoder_hidden_states = floats_tensor((batch_size, 4, 32))

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 4, 32, 32)

    @property
    def output_shape(self):
        return (4, 4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_lora_processors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        model = self.model_class(**init_dict)

        with paddle.no_grad():
            sample1 = model(**inputs_dict).sample

        lora_attn_procs = create_lora_3d_layers(model)

        # make sure we can set a list of attention processors
        model.set_attn_processor(lora_attn_procs)

        # test that attn processors can be set to itself
        model.set_attn_processor(model.attn_processors)

        with paddle.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample1 - sample2).abs().max() < 3e-3
        assert (sample3 - sample4).abs().max() < 3e-3

        # sample 2 and sample 3 should be different
        assert (sample2 - sample3).abs().max() > 3e-3

    def test_lora_save_load(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        paddle.seed(0)
        model = self.model_class(**init_dict)

        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_3d_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(
                tmpdirname,
                to_diffusers=False,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, lora_weights_name)))
            paddle.seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, from_diffusers=False)

        with paddle.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample - new_sample).abs().max() < 1e-3

        # LoRA and no LoRA should NOT be the same
        assert (sample - old_sample).abs().max() > 1e-4

    def test_lora_save_load_safetensors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        paddle.seed(0)
        model = self.model_class(**init_dict)

        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_3d_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, safe_serialization=True, to_diffusers=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            paddle.seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, use_safetensors=True, from_diffusers=True)

        with paddle.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample - new_sample).abs().max() < 3e-3

        # LoRA and no LoRA should NOT be the same
        assert (sample - old_sample).abs().max() > 1e-4

    # def test_lora_save_safetensors_load_torch(self):
    #     # enable deterministic behavior for gradient checkpointing
    #     init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

    #     init_dict["attention_head_dim"] = 8

    #     paddle.seed(0)
    #     model = self.model_class(**init_dict)

    #     lora_attn_procs = create_lora_layers(model, mock_weights=False)
    #     model.set_attn_processor(lora_attn_procs)
    #     # Saving as paddle, properly reloads with directly filename
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         model.save_attn_procs(tmpdirname, to_diffusers=True)
    #         self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
    #         paddle.seed(0)
    #         new_model = self.model_class(**init_dict)
    #         new_model.load_attn_procs(
    #             tmpdirname, weight_name="pytorch_lora_weights.bin", use_safetensors=False, from_diffusers=True
    #         )

    def test_lora_save_paddle_force_load_safetensors_error(self):
        pass

    def test_lora_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        paddle.seed(0)
        model = self.model_class(**init_dict)

        with paddle.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_3d_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with paddle.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample

        model.set_attn_processor(AttnProcessor())

        with paddle.no_grad():
            new_sample = model(**inputs_dict).sample

        assert (sample - new_sample).abs().max() < 1e-4
        assert (sample - old_sample).abs().max() < 3e-3

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_lora_xformers_on_off(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 4

        paddle.seed(0)
        model = self.model_class(**init_dict)
        lora_attn_procs = create_lora_3d_layers(model)
        model.set_attn_processor(lora_attn_procs)

        # default
        with paddle.no_grad():
            sample = model(**inputs_dict).sample

            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample

            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample

        assert (sample - on_sample).abs().max() < 0.005
        assert (sample - off_sample).abs().max() < 0.005


@slow
@require_paddle_gpu
class LoraIntegrationTests(unittest.TestCase):
    def test_dreambooth_old_format(self):
        generator = paddle.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/lora_dreambooth_dog_example"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id)
        images = pipe(
            "A photo of a sks dog floating in the river", output_type="np", generator=generator, num_inference_steps=5
        ).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.8065, 0.8114, 0.7948, 0.822, 0.8092, 0.8031, 0.8052, 0.8012, 0.7677])
        self.assertTrue(np.allclose(images, expected, atol=0.0001))

    def test_dreambooth_text_encoder_new_format(self):
        generator = paddle.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/lora-trained"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id)
        images = pipe("A photo of a sks dog", output_type="np", generator=generator, num_inference_steps=5).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3782, 0.367, 0.3792, 0.3758, 0.3719, 0.3851, 0.382, 0.3842, 0.4009])
        self.assertTrue(np.allclose(images, expected, atol=0.0001))

    def test_a1111(self):
        generator = paddle.Generator().manual_seed(0)
        # hf-internal-testing/Counterfeit-V2.5 -> gsdf/Counterfeit-V2.5
        pipe = StableDiffusionPipeline.from_pretrained("gsdf/Counterfeit-V2.5", safety_checker=None)
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, from_diffusers=True)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array(
            [0.5849, 0.5824, 0.5898, 0.5802, 0.5676, 0.5974, 0.5921, 0.5987, 0.6081],
        )

        self.assertTrue(np.allclose(images, expected, atol=0.0001))

    def test_a1111_with_model_cpu_offload(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_a1111_with_sequential_cpu_offload(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_kohya_sd_v15_with_higher_dimensions(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        lora_model_id = "hf-internal-testing/urushisato-lora"
        lora_filename = "urushisato_v15.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 0.6871, 0.6361, 0.5694])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_vanilla_funetuning(self):
        generator = paddle.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/sd-model-finetuned-lora-t4"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id, from_diffusers=True)
        images = pipe("A pokemon with blue eyes.", output_type="np", generator=generator, num_inference_steps=5).images
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array(
            [0.719, 0.716, 0.6854, 0.7449, 0.7188, 0.7124, 0.7403, 0.7287, 0.6979],
        )
        self.assertTrue(np.allclose(images, expected, atol=0.0001))

    def test_unload_kohya_lora(self):
        generator = paddle.Generator().manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 5
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()
        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, from_diffusers=True)
        generator = paddle.Generator().manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()
        pipe.unload_lora_weights()
        generator = paddle.Generator().manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()
        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=0.001))

    def test_load_unload_load_kohya_lora(self):
        # This test ensures that a Kohya-style LoRA can be safely unloaded and then loaded
        # without introducing any side-effects. Even though the test uses a Kohya-style
        # LoRA, the underlying adapter handling mechanism is format-agnostic.
        generator = paddle.Generator().manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 5
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, from_diffusers=True)
        generator = paddle.Generator().manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = paddle.Generator().manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=0.001))
        # make sure we can load a LoRA again after unloading and they don't have
        # any undesired effects.
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, from_diffusers=True)
        generator = paddle.Generator().manual_seed(0)
        lora_images_again = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()
        self.assertTrue(np.allclose(lora_images, lora_images_again, atol=0.001))

    def test_sdxl_0_9_lora_one(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-daiton-lora"
        lora_filename = "daiton-xl-lora-test.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3838, 0.3482, 0.3588, 0.3162, 0.319, 0.3369, 0.338, 0.3366, 0.3213])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_0_9_lora_two(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-costumes-lora"
        lora_filename = "saijo.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3137, 0.3269, 0.3355, 0.255, 0.2577, 0.2563, 0.2679, 0.2758, 0.2626])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_0_9_lora_three(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-kamepan-lora"
        lora_filename = "kame_sdxl_v2-000020-16rank.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4015, 0.3761, 0.3616, 0.3745, 0.3462, 0.3337, 0.3564, 0.3649, 0.3468])

        self.assertTrue(np.allclose(images, expected, atol=5e-3))

    def test_sdxl_1_0_lora(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_sdxl_1_0_lora_fusion(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        # This way we also test equivalence between LoRA fusion and the non-fusion behaviour.
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_sdxl_1_0_lora_unfusion(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_with_fusion = images[0, -3:, -3:, -1].flatten()

        pipe.unfuse_lora()
        generator = paddle.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_without_fusion = images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(images_with_fusion, images_without_fusion, atol=1e-3))

    def test_sdxl_1_0_lora_unfusion_effectivity(self):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()

        generator = paddle.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        original_image_slice = images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        generator = paddle.Generator().manual_seed(0)
        _ = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        pipe.unfuse_lora()
        generator = paddle.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_without_fusion_slice = images[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(original_image_slice, images_without_fusion_slice, atol=1e-3))

    def test_sdxl_1_0_lora_fusion_efficiency(self):
        generator = paddle.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        start_time = time.time()
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_non_fusion = end_time - start_time

        del pipe

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        start_time = time.time()
        generator = paddle.Generator().manual_seed(0)
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_fusion = end_time - start_time

        self.assertTrue(elapsed_time_fusion < elapsed_time_non_fusion)

    def test_sdxl_1_0_last_ben(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "TheLastBen/Papercut_SDXL"
        lora_filename = "papercut.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe("papercut.safetensors", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.5244, 0.4347, 0.4312, 0.4246, 0.4398, 0.4409, 0.4884, 0.4938, 0.4094])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_1_0_fuse_unfuse_all(self):
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        text_encoder_1_sd = copy.deepcopy(pipe.text_encoder.state_dict())
        text_encoder_2_sd = copy.deepcopy(pipe.text_encoder_2.state_dict())
        unet_sd = copy.deepcopy(pipe.unet.state_dict())

        pipe.load_lora_weights(
            "davizca87/sun-flower", weight_name="snfw3rXL-000004.safetensors", paddle_dtype=paddle.float16
        )
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        pipe.unfuse_lora()

        assert state_dicts_almost_equal(text_encoder_1_sd, pipe.text_encoder.state_dict())
        assert state_dicts_almost_equal(text_encoder_2_sd, pipe.text_encoder_2.state_dict())
        assert state_dicts_almost_equal(unet_sd, pipe.unet.state_dict())

    def test_sdxl_1_0_lora_with_sequential_cpu_offloading(self):
        generator = paddle.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))
