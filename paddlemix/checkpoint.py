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

import os
import shutil

import paddle
import paddle.nn.functional as F


def save(args, model, optimizer, epoch=0, step=0, output_dir="", is_best=False):
    """
    save the state dicts of model and optimizer into an checkpoint.
    """
    if args.dp_rank != 0:
        return

    if output_dir and isinstance(output_dir, str):
        output_dir = os.path.join(output_dir, "epoch_%d_step_%d" % (epoch, step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        print("Save model to %s" % output_dir)

        save_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}".format(output_dir, args.mp_rank, args.sharding_rank)

        # if args.sharding_stage == 3:
        #     model.get_all_parameters(convert2cpu=False)
        paddle.save(model.state_dict(), os.path.join(save_dir, "model.pdparams"))
        paddle.save(optimizer.state_dict(), os.path.join(save_dir, "model_state.pdopt"))
        if is_best:
            shutil.copyfile("model.pdparams", "model_best.pdparams")
        meta_dict = {
            "epoch": epoch,
            "step": step,
            "cuda_rng_state": paddle.get_cuda_rng_state(),
        }
        paddle.save(meta_dict, os.path.join(save_dir, "meta_state.pdopt"))

    else:
        raise TypeError("`save` requires a valid value of `output_dir`.")


def load_model(args, model, optimizer=None, ckpt_dir=""):
    """
    load the saved checkpoint file and update the state dicts of model and optimizer.
    """
    if ckpt_dir and isinstance(ckpt_dir, str) and os.path.isdir(ckpt_dir):
        print("Try to load checkpoint from %s " % ckpt_dir)

        load_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}".format(ckpt_dir, args.mp_rank, args.sharding_rank)
        model_path = os.path.join(load_dir, "model.pdparams")
        opt_path = os.path.join(load_dir, "model_state.pdopt")
        # meta_path = os.path.join(load_dir, "meta_state.pdopt")

        if os.path.exists(model_path):
            model_dict = paddle.load(model_path)
            for name, param in model.state_dict().items():
                assert name in model_dict.keys(), "No param named `{}` was found in checkpoint file.".format(name)

                if param.dtype != model_dict[name].dtype:
                    model_dict[name] = model_dict[name].cast(param.dtype)

            model.set_state_dict(model_dict)
            del model_dict
        else:
            raise ValueError("No checkpoint file found in %s" % model_path)

        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
            del opt_dict
        else:
            print("No optimizer checkpoint file found in %s." % opt_path)

        # if os.path.exists(meta_path):
        #     meta_dict = paddle.load(meta_path)
        #     load_recovery = {
        #         'step': meta_dict['step'],
        #         'epoch': meta_dict['epoch'],
        #         'rng_state': meta_dict['cuda_rng_state']
        #     }
        #     del meta_dict
        # else:
        #     raise ValueError("No meta checkpoint file found in %s." %
        #                         meta_path)

        print("successfully load checkpoints")
    elif ckpt_dir and os.path.isfile(ckpt_dir):
        print("Try to load a whole checkpoint from %s " % ckpt_dir)
        embedding_list = ["token_embedding"]
        collinear_list = [
            "proj",
            "w1",
            "w2",
            "w3",
            "head",
            "c_fc",
            "c_proj",
            "q_bias",
            "v_bias",
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv",
            "c_fc",
            "c_proj",
            "lm_head",
            "fc1",
            "fc2",
            "fc3",
        ]
        rowlinear_list = ["out_proj"]  # in eva_text_model.py, but evaclip do not use text model
        all_list = collinear_list + rowlinear_list + embedding_list
        skip_list = [
            "visual.patch_embed.proj.weight",
            "visual.patch_embed.proj.bias",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
        ]

        col_list = []
        row_list = []
        emb_list = []

        mp_rank = args.mp_rank
        mp_size = max(args.tensor_parallel_degree, 1)

        def col_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[1] // mp_size
                return model_dict[:, mp_rank * subbatch : (mp_rank + 1) * subbatch]
            elif len(model_dict.shape) == 1:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        def row_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]
            else:
                return model_dict

        def emb_split_modeldict(model_dict):
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        model_dict = paddle.load(ckpt_dir)
        modelkeys = model_dict.keys()
        for whole_key in modelkeys:
            if "." not in whole_key:
                continue

            key = whole_key.split(".")[-2]
            if whole_key in skip_list:
                continue
            if key in all_list:
                if key in collinear_list:
                    col_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = col_split_modeldict(model_dict[whole_key])
                elif key in rowlinear_list:
                    row_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = row_split_modeldict(model_dict[whole_key])
                else:
                    emb_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = emb_split_modeldict(model_dict[whole_key])

        if hasattr(args, "context_length") and args.context_length != 77:
            model_dict["text.positional_embedding"] = model_dict["text.positional_embedding"][: args.context_length, :]

        # interpolate position embedding, only in eva02 finetune large size training
        if "pos_embed" in model_dict and hasattr(model, "patch_embed"):
            pos_embed_checkpoint = model_dict["pos_embed"]  #
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]).transpose(
                    perm=[0, 3, 1, 2]
                )
                pos_tokens = F.interpolate(
                    pos_tokens.astype(dtype="float32"), size=(new_size, new_size), mode="bicubic", align_corners=False
                )
                pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(start_axis=1, stop_axis=2)
                new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
                model_dict["pos_embed"] = new_pos_embed

        print("cast state_dict to default dtype:{}".format(paddle.get_default_dtype()))
        for key, value in model_dict.items():
            if "freqs_cos" in key or "freqs_sin" in key:
                continue
            model_dict[key] = paddle.cast(value, dtype=paddle.get_default_dtype())
        model.set_state_dict(model_dict)
        del model_dict
    else:
        print("`load` requires a valid value of `ckpt_dir`.")
        raise TypeError("`load` requires a valid value of `ckpt_dir`.")
