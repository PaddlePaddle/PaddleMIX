## Grounded-SAM: Detect and Segment Everything with Text Prompt

## Prepare

```bash
# install
pip install paddlenlp 

#Multi-scale deformable attention custom OP compilation
cd /paddlevlp/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```


## dynamic inference
```bash

python grounded-sam.py \
--input_image image_you_want_to_detect.jpg \
--prompt eyes
--dino_model_name_or_path GroundingDino/groundingdino-swint-ogc\
--sam_model_name_or_path Sam/SamVitH
```