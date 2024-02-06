# Stable Video Diffusion image-to-video fine-tuning

The `train_image_to_video_svd.py` script shows how to fine-tune Stable Video Diffusion (SVD) on your own dataset.

🚨 This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparamters to get the best result on your dataset. 🚨

## Running locally with Paddle

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX/ppdiffusers
pip install -e .
```

Then cd in the `examples/stable_video_diffusion` folder and run
```bash
pip install -r requirements_svd.txt
```

### Video Data Processing
We will use BDD100K as an example for training data processing. Note that BDD100K is a driving video/image dataset, but this is not a necessity for training. Any video can be used to initiate your training. Please refer to the `DummyDataset` data reading logic. In short, you only need to specify `--train_data_dir` and `--valid_data_path`. Then arrange your videos in the following file structure:
```bash
self.base_folder
    ├── video_name1
    │   ├── video_frame1
    │   ├── video_frame2
    │   ...
    ├── video_name2
    │   ├── video_frame1
        ├── ...
```

### 数据准备
Execute the following command to download and extract the processed dataset.
```
wget https://paddlenlp.bj.bcebos.com/models/community/westfish/lvdm_datasets/sky_timelapse_lvdm.zip && unzip sky_timelapse_lvdm.zip
wget https://example.com/dataset.zip && unzip dataset.zip

```

### Training
#### Single machine, single GPU, V100 32G
```bash
export MODEL_NAME="stabilityai/stable-video-diffusion-img2vid-xt"
export DATASET_NAME="bdd100k"
export OUTPUT_DIR="sdv_train_output"
export VALID_DATA="valid_image"
export GLOG_minloglevel=2

python train_image_to_video_svd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_NAME \
    --valid_data_path=$VALID_DATA \
    --width=448 --height=256 --enable_xformers_memory_efficient_attention --gradient_checkpointing
```

#### Single machine, single GPU
```bash
export MODEL_NAME="stabilityai/stable-video-diffusion-img2vid-xt"
export DATASET_NAME="bdd100k"
export OUTPUT_DIR="sdv_train_output"
export VALID_DATA="valid_image"
export GLOG_minloglevel=2
export FLAGS_conv_workspace_size_limit=4096

python train_image_to_video_svd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_NAME \
    --valid_data_path=$VALID_DATA
```

#### Single machine, multiple GPUs
```bash
export MODEL_NAME="stabilityai/stable-video-diffusion-img2vid-xt"
export DATASET_NAME="bdd100k"
export OUTPUT_DIR="sdv_train_output"
export VALID_DATA="valid_image"
export GLOG_minloglevel=2
export FLAGS_conv_workspace_size_limit=4096

python train_image_to_video_svd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_NAME \
    --valid_data_path=$VALID_DATA
```
**Notes**:

*  "bf16" only supported on NVIDIA A100.

### Inference

```python
import paddle

from ppdiffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from ppdiffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "your-stable-video-diffusion-img2vid-model-path-or-id",
    paddle_dtype=paddle.float16
)

# Load the conditioning image
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=")
image = load_image("rocket.png")
image = image.resize((1024, 576))

generator = paddle.Generator().manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

### Comparison
```python
size=(512, 320), motion_bucket_id=127, fps=7, noise_aug_strength=0.00
generator=torch.manual_seed(111)
```
| Init Image        | Before Fine-tuning |After Fine-tuning |
|---------------|-----------------------------|-----------------------------|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/1587c4b5-c104-4d22-8d56-c86e8c716b06)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/18b5af34-d38f-4d19-8856-77895466d152)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/c464397e-aa05-4d8e-9563-3cc78ad04cb3)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/af3bd957-5b8e-4c21-8791-c9a295761973)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/26d38418-b6fa-40a5-afa6-b278d088638f)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/a49264da-6ccf-48d7-914f-8b0fff9bc99e)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/2a761c41-d6b2-48b8-a63c-505780369484)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/579bed68-2b31-45d5-8cf2-a4e768fec126)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/eaffe1d5-999b-4d27-8d77-d8e8fd1cd380)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/09619a6e-50a2-4aec-afb7-d34c071da425)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/2e525ede-474e-499a-9bc5-8f60700ca3fb)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/ec77f39f-653a-4fa7-8ac0-68f8512f9ddb)|


### References
* https://github.com/pixeli99/SVD_Xtend
