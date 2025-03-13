fuser -k /dev/nvidia*

#sh paddlemix/examples/qwen2_vl/shell/baseline_2b_bs32_1e8.sh
#sh paddlemix/examples/qwen2_vl/shell/baseline_2b_lora_bs32_1e8.sh
#sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_bs32_1e8.sh
#sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_lora_bs32_1e8.sh


#sh paddlemix/examples/qwen2_vl/shell/baseline_7b_bs32_1e8.sh
#sh paddlemix/examples/qwen2_vl/shell/baseline_7b_lora_bs32_1e8.sh
#sh paddlemix/examples/qwen2_5_vl/shell/baseline_7b_bs32_1e8.sh
#sh paddlemix/examples/qwen2_5_vl/shell/baseline_7b_lora_bs32_1e8.sh




#sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_tiny_lora_bs16_1e5.sh
#sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_tiny_sft_bs16_1e5.sh

sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_small_lora_bs16_1e5.sh
#sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_small_sft_bs16_1e5.sh





# python paddlemix/examples/deepseek_vl2/single_image_infer.py \
#     --model_path="deepseek-ai/deepseek-vl2-small" \
#     --image_file="paddlemix/demo_images/examples_image2.jpg" \
#     --question="The Panda" \
#     --dtype="bfloat16"




#sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh

# sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh

#sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh


#sh paddlemix/examples/internvl2/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full.sh
#sh paddlemix/examples/internvl2/shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_full.sh
#sh paddlemix/examples/internvl2/shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_full.sh
#sh paddlemix/examples/internvl2/shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_full.sh

fuser -k /dev/nvidia*

