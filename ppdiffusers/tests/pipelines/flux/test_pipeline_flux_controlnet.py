import unittest

import numpy as np
import paddle

from PIL import Image

from ppdiffusers.transformers import (
    AutoTokenizer, 
    CLIPTextConfig, 
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPVisionConfig, 
    CLIPVisionModelWithProjection,
    T5EncoderModel
)

from ppdiffusers import (
    AutoencoderKL, 
    FlowMatchEulerDiscreteScheduler, 
    FluxControlNetModel,
    FluxMultiControlNetModel,
    FluxControlNetPipeline, 
    FluxTransformer2DModel
)

from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


class FluxControlNetPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds", "control_image", "controlnet_conditioning_scale"])
    batch_params = frozenset(["prompt", "control_image"])

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        paddle.seed(seed=0)
        
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        paddle.seed(seed=0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        paddle.seed(seed=0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        # Create a dummy controlnet
        paddle.seed(seed=0)
        controlnet = FluxControlNetModel(
            transformer_config=transformer.config,
        )

        paddle.seed(seed=0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        # Create dummy image encoder for IP-Adapter
        clip_vision_config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
        )
        image_encoder = CLIPVisionModelWithProjection(clip_vision_config)

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "controlnet": controlnet,
            "vae": vae,
            "image_encoder": image_encoder,
        }

    def get_dummy_inputs(self, seed=0):
        paddle.seed(seed=seed)

        control_image = Image.new("RGB", (16, 16), 0)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "control_image": control_image,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "controlnet_conditioning_scale": 0.8,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_flux_controlnet_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components())

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-6

    def test_flux_controlnet_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        inputs = self.get_dummy_inputs()

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            max_sequence_length=inputs["max_sequence_length"],
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

    def test_flux_controlnet_conditioning_scale(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        
        # Generate with default conditioning scale
        inputs = self.get_dummy_inputs()
        default_output = pipe(**inputs).images[0]
        
        # Generate with higher conditioning scale
        inputs = self.get_dummy_inputs()
        inputs["controlnet_conditioning_scale"] = 1.5
        high_scale_output = pipe(**inputs).images[0]
        
        # Images should be different with different conditioning scales
        max_diff = np.abs(default_output - high_scale_output).max()
        assert max_diff > 1e-6

    def test_flux_controlnet_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_flux_multi_controlnet(self):
        components = self.get_dummy_components()
        
        # Replace the single controlnet with a multi-controlnet
        paddle.seed(seed=0)
        controlnet1 = FluxControlNetModel(
            transformer_config=components["transformer"].config,
        )
        
        paddle.seed(seed=0)
        controlnet2 = FluxControlNetModel(
            transformer_config=components["transformer"].config,
        )
        
        components["controlnet"] = FluxMultiControlNetModel([controlnet1, controlnet2])
        
        pipe = self.pipeline_class(**components)
        
        # Create two different control images
        control_image1 = Image.new("RGB", (16, 16), 0)
        control_image2 = Image.new("RGB", (16, 16), 128)
        
        # Test with multiple control images
        inputs = self.get_dummy_inputs()
        inputs["control_image"] = [control_image1, control_image2]
        inputs["controlnet_conditioning_scale"] = [0.8, 0.7]
        
        output = pipe(**inputs).images[0]
        
        # Just verify it runs without errors
        assert output.shape[-1] == 3
