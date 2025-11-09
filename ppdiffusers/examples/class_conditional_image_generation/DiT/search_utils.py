from typing import List
import paddle
import numpy as np
import random
import pathlib
import os
import copy

# from opensora.schedulers.dpms.dpm_solver import NoiseScheduleVP, get_named_beta_schedule

# betas = torch.tensor(get_named_beta_schedule("linear", 1000))
# noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)

# def get_time_steps(skip_type, t_T, t_0, N, device):
#     """Compute the intermediate time steps for sampling.

#     Args:
#         skip_type: A `str`. The type for the spacing of the time steps. We support three types:
#             - 'logSNR': uniform logSNR for the time steps.
#             - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
#             - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
#         t_T: A `float`. The starting time of the sampling (default is T).
#         t_0: A `float`. The ending time of the sampling (default is epsilon).
#         N: A `int`. The total number of the spacing of the time steps.
#         device: A torch device.
#     Returns:
#         A pytorch tensor of the time steps, with the shape (N + 1,).
#     """
#     if skip_type == "logSNR":
#         lambda_T = noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
#         lambda_0 = noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
#         logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
#         return noise_schedule.inverse_lambda(logSNR_steps)
#     elif skip_type == "time_uniform":
#         return torch.linspace(t_T, t_0, N + 1).to(device)
#     elif skip_type == "time_quadratic":
#         t_order = 2
#         return torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
#     else:
#         raise ValueError(
#             f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'"
#         )

def get_time_steps(num_inference_steps, timestep_spacing):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
    """
    # Clipping the minimum of all lambda(t) for numerical stability.
    # This is critical for cosine (squaredcos_cap_v2) noise schedule.
    c = paddle.to_tensor(float("-inf"), dtype=paddle.float32)
    lambda_t = paddle.load("/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/lambda_t.pdparams")
    t = paddle.flip(lambda_t, [0])
    clipped_idx = paddle.searchsorted(t, c)
    if paddle.isinf(c):
        clipped_idx = paddle.to_tensor(0)
    # clipped_idx = paddle.searchsorted(paddle.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
    last_timestep = ((1000 - clipped_idx).numpy()).item()

    # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891


    if timestep_spacing == "linspace":
        timesteps = (
            np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        )
    elif timestep_spacing == "leading":
        step_ratio = last_timestep // (num_inference_steps + 1)
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
        timesteps += 0
    elif timestep_spacing == "trailing":
        step_ratio = 1000 / num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
        timesteps -= 1
    else:
        raise ValueError(
            "is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
        )
    return timesteps.tolist()



def get_population(base_path, search_cfg):
    base_path = pathlib.Path(base_path)
    data_paths = [file for file in base_path.glob('**/*.{}'.format("pdparams"))]
    population = []
    for path in data_paths:
        population.append(paddle.load(str(path)))
    print(f"Population size: {len(population)}")
    population = sorted(population, key=lambda x: x[1] * search_cfg["metric"].get("indicator", 1))

    return population

def select_parents(population, search_cfg, num=1):
    # import ipdb;ipdb.set_trace()
    if random.uniform(0, 1) < search_cfg["parents"]["rank_prob"]:
        mode = "rank"
    else:
        mode = "absolute"
        
    if mode == "rank":
        all_candidate_parents = population[:search_cfg["parents"]["rank_bar"]]
    elif mode == "absolute":
        all_candidate_parents = []
        for p in population[1:]:
            if abs(p[1] - population[0][1]) < search_cfg["parents"]["absolute_bar"]:
                all_candidate_parents.append(p)
            else:
                break
    
    selected_parents = random.sample(all_candidate_parents, num)
    if num == 1:
        return selected_parents[0]
    else:
        return selected_parents

def get_baseline(cfg, budget, data_path):
    for timestep_type in cfg["baseline"]["timestep_type"]:
        for order in cfg["baseline"]["orders"]:
            prefix = f"{budget}step_{timestep_type}_order{order}"
            if os.path.exists(os.path.join(data_path, "baselines", f"{prefix}.occ")) or \
                os.path.exists(os.path.join(data_path, "baselines", f"{prefix}.pdparams")):
                continue
            paddle.save("occ", os.path.join(data_path, "baselines", f"{prefix}.occ"))

            # get order list
            order_list = [order] * budget
            for i in range(order):
                order_list[i] = min(i+1, order)
                # import ipdb; ipdb.set_trace()
            # get timesteps

            timesteps = get_time_steps(num_inference_steps = budget, timestep_spacing=timestep_type)

            return {"orders": order_list, "timesteps": timesteps}, os.path.join(data_path, "baselines", f"{prefix}.pdparams")
    
    return -1, -1

def crossover(parents_1, parents_2, cfg):
    # import ipdb; ipdb.set_trace()
    new_order_list = []
    for i in range(len(parents_1["orders"])):
        if random.uniform(0, 1) < cfg["crossover"]["better_prob"]:
            new_order_list.append(parents_1["orders"][i])
        else:
            new_order_list.append(parents_2["orders"][i])

    new_timesteps = []
    for i in range(len(parents_1["timesteps"])):
        if random.uniform(0, 1) < cfg["crossover"]["better_prob"]:
            new_timesteps.append(parents_1["timesteps"][i])
        else:
            new_timesteps.append(parents_2["timesteps"][i])

    new_timesteps, _ = paddle.sort(paddle.to_tensor(new_timesteps), descending=True)
    new_timesteps = new_timesteps.cpu().tolist()
    
    return {"orders": new_order_list, "timesteps": new_timesteps}

def mutate(parent, cfg):
    # import ipdb; ipdb.set_trace()
    new_order_list = copy.deepcopy(parent["orders"])
    new_timesteps = copy.deepcopy(parent["timesteps"])

    
    for i in range(len(new_order_list)):
        a = random.uniform(0, 1)
        # import ipdb; ipdb.set_trace()
        print(a)
        if a < cfg["mutate"]["order"]["prob"]:
            order_dist = cfg["mutate"]["order"]["dist"]
            new_order_list[i] = random.choices(list(order_dist.keys()), weights=list(order_dist.values()))[0]
        if i != 0:
            if random.uniform(0, 1) < cfg["mutate"]["timestep"]["prob"]:
                new_timesteps[i] = new_timesteps[i] + int(round(random.gauss(0.0, 1.0) * cfg["mutate"]["timestep"]["scale"]))
        min_time_step =  0
        if new_timesteps[i] < min_time_step:
            new_timesteps[i] = round(random.uniform(min_time_step, new_timesteps[i-1]))
        # new_timesteps[i] = max(1.0 / noise_schedule.total_N + 5e-4, new_timesteps[i])
        new_order_list[i] = min(i+1, new_order_list[i])

    new_timesteps = paddle.sort(paddle.to_tensor(new_timesteps), descending=True)
    new_timesteps = new_timesteps.cpu().tolist()
    
    return {"orders": new_order_list, "timesteps": new_timesteps}
        