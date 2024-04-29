# GLOG_v=2 python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 0 1 --use_new_transpose 0 0 0 1 > log_transfer_layout_unet 2>&1
# GLOG_v=2 python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 0 1 --use_new_transpose 0 0 0 0 > log_no_transfer_layout_unet 2>&1

# GLOG_v=2 python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 1 0 --use_new_transpose 0 0 1 0 > log_transfer_layout_vae_decoder 2>&1
# GLOG_v=2 python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 1 0 --use_new_transpose 0 0 0 0 > log_no_transfer_layout_vae_decoder 2>&1


CUDNN_LOGDEST_DBG=stdout python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 1 0 --use_new_transpose 0 0 1 0 > log_transfer_layout_vae_decoder_with_cudnn_log 2>&1
CUDNN_LOGDEST_DBG=stdout python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name text2img --use_pir 0 0 1 0 --use_new_transpose 0 0 0 0 > log_no_transfer_layout_vae_decoder_with_cudnn_log 2>&1