LOG='fp16_1024'
CFG='w8a8.yaml'
PROMPT_PATH='./samples_16.txt'
GPU_ID=2

# CUDA_VISIBLE_DEVICES=$GPU_ID python get_calib_data.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"  --prompt $PROMPT_PATH

# CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"
CUDA_VISIBLE_DEVICES=$GPU_ID python quant_inference.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"