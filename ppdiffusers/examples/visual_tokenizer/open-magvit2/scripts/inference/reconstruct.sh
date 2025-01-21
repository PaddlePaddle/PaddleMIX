#GPU
python reconstruct.py \
--config_file "configs/gpu/imagenet_lfqgan_256_L.yaml" \
--ckpt_path  ./imagenet_256_L.pdparams \
--save_dir "./visualize" \
--version  "1k" \
--image_num 0 \
--image_size 256  