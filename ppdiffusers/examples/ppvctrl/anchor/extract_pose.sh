python anchor/extract_pose.py  \
       --det_model_dir=./anchor/checkpoints/PP-YOLOE_plus-S_infer \
       --keypoint_model_dir=./anchor/checkpoints/paddle3.0_hrnet_w48_coco_wholebody_384x288/ \
       --video_file=./examples/pose/case1/pixel_values.mp4  \
       --device=gpu \
       --output_dir=./examples/pose/case1/guide_values.mp4 \
       --reference_image_path=./examples/pose/case1/reference_image.jpg \