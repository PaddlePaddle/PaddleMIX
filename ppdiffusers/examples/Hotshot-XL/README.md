转换模型前需要运行 convert_pre.sh，包括 clone 模型，配置修改，torch 关联代码复制
convert_hotshotxl_to_ppdiffusers.py 转换hotshot模型
convert_controlnet.py 转换controlnet模型

使用 develop 版本 paddlepaddle，2.6版本会有\_to转换错误paddle.Place不能识别

### Text-to-GIF

``` shell
python inference.py --pretrained_path model_path \
  --prompt="a bulldog in the captains chair of a spaceship, hd, high quality" \
  --output="output.gif"
```

### Text-to-GIF with ControlNet

``` shell
python inference.py --pretrained_path model_path \
  --prompt="a girl jumping up and down and pumping her fist, hd, high quality" \
  --output="output.gif" \
  --control_type="depth" \
  --gif="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXNneXJicG1mOHJ2dzQ2Y2JteDY1ZWlrdjNjMjl3ZWxyeWFxY2EzdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YOTAoXBgMCmFeQQzuZ/giphy.gif"
```

https://github.com/hotshotco/Hotshot-XL
