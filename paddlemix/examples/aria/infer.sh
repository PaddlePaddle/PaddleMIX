CUDA_VISIBLE_DEVICES=0 python inference.py \
                              --dtype bfloat16 \
                              --base_model_path path_to_Aria \
                              --tokenizer_path path_to_Aria  \
                              --image_path ../demo_images/examples_image1.jpg \
                              --prompt "what is the image?"
