# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#GPU
python generate.py \
--ckpt "./AR_256_XL.pdparams" \
-o "./visualize" \
--config "configs/gpu/imagenet_conditional_llama_XL.yaml" \
-k "0,0" \
-p "0.96,0.96" \
--token_factorization \
-n 1 \
-t "1.0,1.0" \
--classes "207" \
--batch_size 8 \
--cfg_scale "4.0,4.0" \