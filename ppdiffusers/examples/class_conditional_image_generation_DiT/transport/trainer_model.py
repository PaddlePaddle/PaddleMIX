import contextlib
import inspect
import os
import json
import numpy as np
import paddle
import paddle.nn as nn

from ppdiffusers import AutoencoderKL, DDIMScheduler, is_ppxformers_available
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params
from paddlenlp.utils.log import logger

from .sit import DiT
from .transport import Transport, ModelType, WeightType, PathType, Sampler
from . import path
from .utils import mean_flat
from .integrators import ode, sde


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class SiTDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        self.model_mean_type = "epsilon" # PREVIOUS_X START_X EPSILON
        self.model_var_type = "learned_range" # LEARNED FIXED_SMALL FIXED_LARGE LEARNED_RANGE
        self.loss_type = "mse" # MSE RESCALED_MSE KL(is_vb) RESCALED_KL(is_vb)

        self.path_type = "Linear" # LINEAR GVP VP
        self.prediction = "velocity" # VELOCITY NOISE SCORE 
        self.model_type = "velocity" #
        self.loss_weight = "None" #
        self.train_eps = 0
        self.sample_eps = 0

        path_choice = {
            "Linear": PathType.LINEAR,
            "GVP": PathType.GVP,
            "VP": PathType.VP,
        }
        path_type = path_choice[self.path_type]

        if self.loss_weight == "velocity":
            loss_type = WeightType.VELOCITY
        elif self.loss_weight == "likelihood":
            loss_type = WeightType.LIKELIHOOD
        else:
            loss_type = WeightType.NONE
        self.loss_type = loss_type

        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.path_sampler = path_options[path_type]()

        # self.use_timesteps = set(use_timesteps)
        # self.timestep_map = []
        # last_alpha_cumprod = 1.0
        # new_betas = []
        # for i, alpha_cumprod in enumerate(self.alphas_cumprod):
        #     if i in self.use_timesteps:
        #         new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
        #         last_alpha_cumprod = alpha_cumprod
        #         self.timestep_map.append(i)

        # betas = get_named_beta_schedule('linear', 1000)

        # # Use float64 for accuracy.
        # betas = np.array(betas, dtype=np.float64)
        # self.betas = betas
        # assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()
        # self.num_timesteps = int(betas.shape[0])

        # alphas = 1.0 - betas
        # self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        # assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)


        self.transformer = DiT(**read_json(model_args.config_file))
        self.transformer_is_pretrained = False

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type

        # self.noise_scheduler = DDPMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     num_train_timesteps=1000,
        #     prediction_type=self.prediction_type,
        # )
        # self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)

        # if model_args.image_logging_steps > 0:
        #     self.eval_scheduler = DDIMScheduler(
        #         beta_start=0.00085,
        #         beta_end=0.012,
        #         beta_schedule="scaled_linear",
        #         num_train_timesteps=1000,
        #         clip_sample=False,
        #         set_alpha_to_one=False,
        #         steps_offset=1,
        #         prediction_type=self.prediction_type,
        #     )
        #     self.eval_scheduler.set_timesteps(model_args.num_inference_steps)

        self.use_ema = model_args.use_ema
        self.noise_offset = model_args.noise_offset
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)
        self.transformer.train()
        self.vae.eval()

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: paddle.Tensor, noise: paddle.Tensor, timesteps: paddle.Tensor) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.transformer.parameters())
            self.model_ema.copy_to(self.transformer)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.transformer)

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = paddle.tensor(z.shape)
        N = paddle.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - paddle.sum(x ** 2) / 2.
        return paddle.vmap(_fn)(z) ###

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = paddle.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = paddle.rand((x1.shape[0],)) * (t1 - t0) + t0
        #t = t.to(x1)
        return t, x0, x1

    def forward(self, latents=None, label_id=None, **kwargs):
        t, x0, x1 = self.sample(latents)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        self.vae.eval()

        model_output = self.transformer(x=xt, t=t, y=label_id)
        B, *_, C = xt.shape
        ### Note
        model_output, _ = paddle.split(model_output, 2, axis=1)
        assert model_output.shape == [B, *xt.shape[1:-1], C]


        # # Get the target for loss depending on the prediction type
        # if self.prediction_type == "epsilon": # default
        #     target = noise
        # elif self.prediction_type == "v_prediction":
        #     target = self.get_velocity(latents, noise, timesteps)
        # else:
        #     raise ValueError(f"Unknown prediction type {self.prediction_type}")

        if self.model_type == "velocity":
            loss = mean_flat(((model_output - ut) ** 2))
        return loss

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        return score_fn

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, max_batch=8, **kwargs):
        self.eval()
        if pixel_values.shape[0] > max_batch:
            pixel_values = pixel_values[:max_batch]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=256,
        width=256,
        eta=0.0,
        class_labels=[1,2,3,4,5,6,7,8],
        guidance_scale=4.0,
        **kwargs,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 8 image
            if input_ids.shape[0] > 8:
                input_ids = input_ids[:8]

            batch_size = input_ids.shape[0]
            latent_channels = self.transformer.in_channels

            latents = paddle.randn((input_ids.shape[0], self.transformer.in_channels, height // 8, width // 8))
            latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

            class_labels = paddle.to_tensor(class_labels).flatten()
            class_null = paddle.to_tensor([1000] * batch_size)
            class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                if guidance_scale > 1:
                    half = latent_model_input[: len(latent_model_input) // 2]
                    latent_model_input = paddle.concat([half, half], axis=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timesteps = t
                if not paddle.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    if isinstance(timesteps, float):
                        dtype = paddle.float32
                    else:
                        dtype = paddle.int64
                    timesteps = paddle.to_tensor([timesteps], dtype=dtype)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None]
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(
                    [
                        latent_model_input.shape[0],
                    ]
                )
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input, timestep=timesteps, class_labels=class_labels_input
                ).sample

                # perform guidance
                if guidance_scale > 1:
                    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    bs = eps.shape[0]
                    # TODO torch.split vs paddle.split
                    cond_eps, uncond_eps = paddle.split(eps, [bs // 2, bs - bs // 2], axis=0)

                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = paddle.concat([half_eps, half_eps], axis=0)

                    noise_pred = paddle.concat([eps, rest], axis=1)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    # TODO torch.split vs paddle.split
                    model_output, _ = paddle.split(
                        noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                    )
                else:
                    model_output = noise_pred

                # compute previous image: x_t -> x_t-1
                latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

            if guidance_scale > 1:
                latents, _ = latent_model_input.chunk(2, axis=0)
            else:
                latents = latent_model_input

            latents = 1 / self.vae.config.scaling_factor * latents
            samples = self.vae.decode(latents).sample
            samples = (samples / 2 + 0.5).clip(0, 1)
            image = samples.transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.transformer.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)

    def set_xformers(self, use_xformers=False):
        if use_xformers:
            if not is_ppxformers_available():
                raise ValueError(
                    'Please run `python -m pip install "paddlepaddle-gpu>=2.5.0.post117" -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html first.'
                )
            else:
                try:
                    attention_op = os.getenv("FLAG_XFORMERS_ATTENTION_OP", "none").lower()

                    if attention_op == "none":
                        attention_op = None

                    self.transformer.enable_xformers_memory_efficient_attention(attention_op)
                    if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                        self.vae.enable_xformers_memory_efficient_attention(attention_op)
                except Exception as e:
                    logger.warn(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
        else:
            if hasattr(self.transformer, "set_default_attn_processor"):
                self.transformer.set_default_attn_processor()
            if hasattr(self.vae, "set_default_attn_processor"):
                self.vae.set_default_attn_processor()

    def set_ema(self, use_ema=False):
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.transformer)
