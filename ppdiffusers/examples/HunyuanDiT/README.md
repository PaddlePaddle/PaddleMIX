# Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding

This repo contains Paddle model definitions, pre-trained weights and inference/sampling code for  Hunyuan-DiT.

### **Chinese-English Bilingual DiT Architecture**
Hunyuan-DiT is a diffusion model in the latent space, as depicted in figure below. Following the Latent Diffusion Model, we use a pre-trained Variational Autoencoder (VAE) to compress the images into low-dimensional latent spaces and train a diffusion model to learn the data distribution with diffusion models. Our diffusion model is parameterized with a transformer. To encode the text prompts, we leverage a combination of pre-trained bilingual (English and Chinese) CLIP and multilingual T5 encoder.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png"  height=450>
</p>


## Comparisons
comprehensive comparison of Hunyuan-DiT with other models.

<p align="center">
<table>
<thead>
<tr>
    <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Text-Image Consistency (%)</th> <th>Excluding AI Artifacts (%)</th> <th>Subject Clarity (%)</th> <th rowspan="2">Aesthetics (%)</th> <th rowspan="2">Overall (%)</th>
</tr>
</thead>
<tbody>
<tr>
    <td>SDXL</td> <td> ✔ </td> <td>64.3</td> <td>60.6</td> <td>91.1</td> <td>76.3</td> <td>42.7</td>
</tr>
<tr>
    <td>PixArt-α</td> <td> ✔</td> <td>68.3</td> <td>60.9</td> <td>93.2</td> <td>77.5</td> <td>45.5</td>
</tr>
<tr>
    <td>Playground 2.5</td> <td>✔</td> <td>71.9</td> <td>70.8</td> <td>94.9</td> <td>83.3</td> <td>54.3</td>
</tr>

<tr>
    <td>SD 3</td> <td>&#10008</td> <td>77.1</td> <td>69.3</td> <td>94.6</td> <td>82.5</td> <td>56.7</td>

</tr>
<tr>
    <td>MidJourney v6</td><td>&#10008</td> <td>73.5</td> <td>80.2</td> <td>93.5</td> <td>87.2</td> <td>63.3</td>
</tr>
<tr>
    <td>DALL-E 3</td><td>&#10008</td> <td>83.9</td> <td>80.3</td> <td>96.5</td> <td>89.4</td> <td>71.0</td>
</tr>
<tr style="font-weight: bold; background-color: #f2f2f2;">
    <td>Hunyuan-DiT</td><td>✔</td> <td>74.2</td> <td>74.3</td> <td>95.4</td> <td>86.6</td> <td>59.0</td>
</tr>
</tbody>
</table>
</p>



## Inference

### Using Command Line

We provide several commands to quick start:

```shell
# Prompt Enhancement + Text-to-Image. Paddle mode
python sample_t2i.py --prompt "渔舟唱晚"

# Only Text-to-Image. Paddle mode
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# Generate an image with other image sizes.
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

```

More example prompts can be found in [example_prompts.txt](example_prompts.txt)

### More Configurations

We list some more useful configurations for easy usage:

|    Argument     |  Default  |                     Description                     |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        The text prompt for image generation         |
| `--image-size`  | 1024 1024 |           The size of the generated image           |
|    `--seed`     |    42     |        The random seed for generating images        |
| `--infer-steps` |    100    |          The number of steps for sampling           |
|  `--negative`   |     -     |      The negative prompt for image generation       |
| `--infer-mode`  |   paddle   |       The inference mode (paddle, fa, or trt)        |
|   `--sampler`   |   ddpm    |    The diffusion sampler (ddpm, ddim, or dpmms)     |
| `--no-enhance`  |   False   |        Disable the prompt enhancement model         |
| `--model-root`  |   ckpts   |     The root directory of the model checkpoints     |
|  `--load-key`   |    ema    | Load the student model or EMA model (ema or module) |





## References
> [**Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**](https://arxiv.org/abs/2405.08748) <br>

## Start History

<a href="https://star-history.com/#PaddlePaddle/PaddleMIX&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMIX&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMIX&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMIX&type=Date" />
 </picture>
</a>
