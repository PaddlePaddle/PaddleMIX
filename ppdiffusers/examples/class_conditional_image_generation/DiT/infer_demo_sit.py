import json
import numpy as np
from PIL import Image
import paddle

from ppdiffusers import AutoencoderKL
from transport.sit import SiT
from transport import create_transport, Sampler

paddle.device.set_device('gpu')

image_size = "256"
vae_model = "stabilityai/sd-vae-ft-ema"
config_file = "config/SiT_XL_patch2.json"


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


model = SiT(**read_json(config_file))
state_dict = "SiT-XL-2-256x256.pdparams"
model.set_dict(paddle.load(state_dict))
model.eval() # important!
vae = AutoencoderKL.from_pretrained(vae_model)
latent_size = 256 // 8


# Set user inputs:
seed = 0 #@param {type:"number"}
paddle.seed(seed)
num_sampling_steps = 25 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = [207] #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}
sampler_type = "SDE" #@param ["ODE", "SDE"]
# Note: ODE not support yet


# Create diffusion object:
transport = create_transport()
sampler = Sampler(transport)

# Create sampling noise:
n = len(class_labels)
z = paddle.randn([n, 4, latent_size, latent_size])
y = paddle.to_tensor(class_labels)

# Setup classifier-free guidance:
z = paddle.concat([z, z], 0)
y_null = paddle.to_tensor([1000] * n)
y = paddle.concat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
if sampler_type == "SDE":
    SDE_sampling_method = "Euler" #@param ["Euler", "Heun"]
    diffusion_form = "linear" #@param ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
    diffusion_norm = 1 #@param {type:"slider", min:0, max:10.0, step:0.1}
    last_step = "Mean" #@param ["Mean", "Tweedie", "Euler"]
    last_step_size = 0.4 #@param {type:"slider", min:0, max:1.0, step:0.01}
    sample_fn = sampler.sample_sde(
        sampling_method=SDE_sampling_method,
        diffusion_form=diffusion_form, 
        diffusion_norm=diffusion_norm,
        last_step_size=last_step_size, 
        num_steps=num_sampling_steps,
    ) 
elif sampler_type == "ODE":
    # default to Adaptive Solver
    ODE_sampling_method = "dopri5" #@param ["dopri5", "euler", "rk4"]
    atol = 1e-6
    rtol = 1e-3
    sample_fn = sampler.sample_ode(
        sampling_method=ODE_sampling_method,
        atol=atol,
        rtol=rtol,
        num_steps=num_sampling_steps
    ) 
samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
samples = vae.decode(samples / 0.18215).sample

samples = (samples - samples.min()) / (samples.max() - samples.min()) * 255
npimg = samples.astype('uint8').numpy()[0]
npimg = np.transpose(npimg, (1, 2, 0))
img = Image.fromarray(npimg)
img.save("result.png")
