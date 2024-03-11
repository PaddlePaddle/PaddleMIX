原仓库 <https://github.com/hotshotco/Hotshot-XL>

转换模型前需要运行 convert_pre.sh，包括 clone 模型，配置修改，torch 关联代码复制
convert_hotshot_xl_to_ppdiffusers.py 转换hotshot模型
convert_controlnet.py 转换controlnet模型

使用 develop 版本 paddlepaddle，2.6版本会有\_to转换错误paddle.Place不能识别

### Text-to-GIF

测试随机数 452，精度 f32

``` shell
python inference.py \
  --prompt="a bulldog in the captains chair of a spaceship, hd, high quality" \
  --seed 452 --precision f32 \
  --output="output.gif"
```

### Text-to-GIF with ControlNet

controlnet 模型路径 co63oc/hotshotxl/controlnet_depth，在 inference.py 中配置

``` shell
python inference.py \
  --prompt="a girl jumping up and down and pumping her fist, hd, high quality" \
  --seed 452 --precision f32 \
  --output="output.gif" \
  --control_type="depth" \
  --gif="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXNneXJicG1mOHJ2dzQ2Y2JteDY1ZWlrdjNjMjl3ZWxyeWFxY2EzdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YOTAoXBgMCmFeQQzuZ/giphy.gif"
```

微调训练使用V100 32G显示显存不足，暂时没有合适硬件测试

``` shell
python fine_tune.py --data_dir data/ --pretrained_model_name_or_path ./hotshot_output/
```
