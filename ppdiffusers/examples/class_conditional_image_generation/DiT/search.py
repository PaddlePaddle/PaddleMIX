import paddle
from paddlenlp.trainer import set_seed
import yaml
import argparse

from ppdiffusers import DDIMScheduler, DiTPipeline, DPMSolverMultistepScheduler
import os
import shutil
import random
from search_utils import *

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def eval_coeff(coeff, cfg, generator ,prompts, pipe, save_dir, fixed_noise):
    print(f"Begin to evaluate {coeff}")
    sample_idx = 0
    
    # Clean and create save directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate samples
    for i in range(0, len(prompts), cfg.batch_size):
        # import ipdb; ipdb.set_trace()
        batch_prompts = prompts[i : i + cfg.batch_size]
        
        # Sample using DiT with custom orders and timesteps
        # samples = scheduler.sample(
        #     model,
        #     text_encoder,
        #     z_size=(vae.out_channels, *latent_size),
        #     prompts=batch_prompts,
        #     device=device,
        #     additional_args=model_args,
        #     orders=coeff["orders"],
        #     timesteps=coeff["timesteps"],
        #     input_noise=fixed_noise[sample_idx:sample_idx+cfg.batch_size],
        # )
        # import ipdb; ipdb.set_trace()
        samples = pipe(class_labels=batch_prompts, num_inference_steps=10, generator=generator, timesteps_list = coeff["timesteps"], order_list = coeff["orders"], fixed_noise = fixed_noise[sample_idx:sample_idx+cfg.batch_size]).images
        # import ipdb; ipdb.set_trace()
        
        # Decode VAE latents to images

        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}.png")
            sample.save(save_path)
            sample_idx += 1
    from paddle_fid.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths(
        [save_dir,"/share/public-nfs/chenqian/var/dataset/imagenet256/VIRTUAL_imagenet256_labeled.npz"], 
        batch_size=256,
        dims=2048,
        num_workers=8,
    )

    return -fid
        


def main(args):
    dtype = paddle.float32
    cfg = args
    # import ipdb; ipdb.set_trace()

    fixed_noise = paddle.load("/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/dit_fixed_noise_B5000.pdparams")

    fixed_noise = fixed_noise["fixed_noise"]
    prompts = load_prompts(cfg.prompt_path)
    prompts = list(map(int, prompts))

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) 
    pipe.scheduler.config.algorithm_type = "dpmsolver"
    pipe.scheduler.config.solver_order = 3 
    generator = paddle.Generator().manual_seed(0)  
    # set data save path
    os.makedirs(os.path.join(cfg.data_path, cfg.split), exist_ok=True)
    os.makedirs(os.path.join(cfg.data_path, "baselines"), exist_ok=True)
    
    # search
    with open(cfg.search_config, "r") as f:
        search_cfg = yaml.safe_load(f)
    baseline_done = False
    data_num = len([d for d in os.listdir(os.path.join(cfg.data_path, cfg.split)) if ".pdparams" in str(d)])
    image_save_dir = os.path.join(cfg.data_path, cfg.split, "images")
    
    # Main search loop
    while(1):
        print(f"Random check: {random.uniform(0, 1)}")
        
        # Baseline evaluation phase
        if not baseline_done:
            baseline, data_path = get_baseline(search_cfg, cfg.budget, cfg.data_path)
            if baseline == -1:
                baseline_done = True
                print("All baselines evaluated, starting evolutionary search...")
            else:
                score = eval_coeff(baseline, cfg, generator, prompts, pipe, image_save_dir, fixed_noise)
                paddle.save([baseline, score], data_path)
                
                print(f"Save baseline {[baseline, score]} to {data_path}")
                
                # delete occ file
                str_data_path = str(data_path)
                assert ".pdparams" in str_data_path
                occ_path = str(data_path).replace(".pdparams", ".occ")
                if os.path.exists(occ_path):
                    os.remove(occ_path)
                
                continue
        
        # Evolutionary search phase
        population = get_population(cfg.data_path, search_cfg)
        
        # Decide between crossover and mutation
        if random.uniform(0, 1) < search_cfg["crossover"]["prob"]:
            # Crossover operation
            parents_1, parents_2 = select_parents(population, search_cfg, num=2)
            print(f"Choose {parents_1} and {parents_2} as the crossover parents")
            new_coeff = crossover(parents_1[0], parents_2[0], search_cfg)
        else:
            new_coeff = None
        
        if new_coeff is not None:
            parent = new_coeff
        else:
            # Mutation operation
            parent = select_parents(population, search_cfg, num=1)[0]
            print(f"Choose {parent} as the mutation parent")
        
        new_coeff = mutate(parent, search_cfg)
        
        # Evaluate new coefficient
        score = eval_coeff(new_coeff, cfg, generator, prompts, pipe, image_save_dir, fixed_noise)
        
        # Save result
        result_path = os.path.join(cfg.data_path, cfg.split, f"{data_num}.pdparams")
        paddle.save([new_coeff, score], result_path)
        print(f"Save {[new_coeff, score]} to {result_path}")
        data_num += 1
        
        # Update the search config (in case it was modified externally)
        try:
            with open(cfg.search_config, "r") as f:
                search_cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not reload search config: {e}")

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--search_config", type=str, default="/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/evo_search.yml")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/search_result_debug")
    parser.add_argument("--split", type=str, default="0")
    parser.add_argument("--prompt_path", type=str, default="/share/chenqian-local/PaddleMIX/ppdiffusers/examples/class_conditional_image_generation/DiT/dit_5k.txt")  
    parser.add_argument("--batch_size", type=int, default=40) 
    args = parser.parse_args()
    
    # import ipdb; ipdb.set_trace()
    main(args)
