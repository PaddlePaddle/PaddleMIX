# import sys
# import paddle
# import math
# import re
# import paddlenlp.transformers as transformers

# def build_vision_tower():
#     # vision_tower = 'openai/clip-vit-large-patch14-336'
#     vision_tower = '/home/ma-user/work/yk/pd/openai/clip-vit-large-patch14-336'
#     return CLIPVisionTower(vision_tower)


# def build_vision_projector():
#     projector_type = 'mlp2x_gelu'
#     mm_hidden_size = 1024
#     hidden_size = 4096
#     mlp_gelu_match = re.match('^mlp(\\d+)x_gelu$', projector_type)
#     if mlp_gelu_match:
#         mlp_depth = int(mlp_gelu_match.group(1))
#         modules = [paddle.nn.Linear(in_features=mm_hidden_size,
#             out_features=hidden_size)]
#         for _ in range(1, mlp_depth):
#             modules.append(paddle.nn.GELU())
#             modules.append(paddle.nn.Linear(in_features=hidden_size,
#                 out_features=hidden_size))
#         return paddle.nn.Sequential(*modules)
#     if projector_type == 'identity':
#         return IdentityMap()
#     raise ValueError(f'Unknown projector type: {projector_type}')


# class IdentityMap(paddle.nn.Layer):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x, *args, **kwargs):
#         return x

#     @property
#     def config(self):
#         return {'mm_projector_type': 'identity'}


# class CLIPVisionTower(paddle.nn.Layer):

#     def __init__(self, vision_tower):
#         super().__init__()
#         self.is_loaded = False
#         self.is_resize_pos = False
#         self.vision_tower_name = vision_tower
#         self.select_layer = -1
#         self.select_feature = 'patch'
#         self.load_model()
#         self.resize_pos()

#     def load_model(self):
#         self.vision_tower = transformers.CLIPVisionModel.from_pretrained(self.vision_tower_name)
#         out_0 = self.vision_tower
#         out_0.stop_gradient = not False
#         self.vision_tower = out_0
#         self.is_loaded = True

#     def resize_pos(self):
#         # pos_embed_checkpoint = (self.vision_tower.vision_model.embeddings.position_embedding.weight)
#         pos_embed_checkpoint = (self.vision_tower.vision_model.positional_embedding.weight)
#         pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(axis=0)
#         orig_size = 24
#         new_size = 16
#         if pos_embed_checkpoint.shape[1] == new_size ** 2 + 1:
#             self.is_resize_pos = True
#         else:
#             embedding_size = pos_embed_checkpoint.shape[-1]
#             num_extra_tokens = 1
#             new_num = new_size ** 2 + num_extra_tokens
#             print('Position interpolate from %dx%d to %dx%d' % (orig_size,
#                 orig_size, new_size, new_size))
#             extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#             pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#             pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size,
#                 embedding_size]).transpose(perm=[0, 3, 1, 2])
#             pos_tokens = paddle.nn.functional.interpolate(x=pos_tokens,
#                 size=(new_size, new_size), mode='bicubic', align_corners=False)
#             pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(
#                 start_axis=1, stop_axis=2)
#             new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
#             new_pos_embed = new_pos_embed.squeeze(axis=0)
#             # (self.vision_tower.vision_model.embeddings.position_embedding) = (
#             self.vision_tower.vision_model.positional_embedding = paddle.nn.Embedding(num_embeddings=new_num, embedding_dim=1024)
#             out_1 = paddle.create_parameter(shape=new_pos_embed.to(
#                 pos_embed_checkpoint.dtype).shape, dtype=new_pos_embed.to(
#                 pos_embed_checkpoint.dtype).numpy().dtype,
#                 default_initializer=paddle.nn.initializer.Assign(
#                 new_pos_embed.to(pos_embed_checkpoint.dtype)))
#             out_1.stop_gradient = not True
#             # (self.vision_tower.vision_model.embeddings.position_embedding.weight) = out_1
#             self.vision_tower.vision_model.positional_embedding.weight = out_1
#             # self.vision_tower.vision_model.embeddings.position_ids = (paddle
#             self.vision_tower.vision_model.position_ids = paddle.arange(end=new_num).expand(shape=(1, -1))
#             self.is_resize_pos = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(
#                 f'Unexpected select feature: {self.select_feature}')
#         return image_features

#     def forward(self, images):
#         if not self.is_loaded:
#             self.load_model()
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.unsqueeze(axis=0), output_hidden_states=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(images, output_hidden_states=True)
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)
#         return image_features

#     @property
#     def dummy_feature(self):
#         return paddle.zeros(shape=[1, self.hidden_size])

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2


# class PLoRA(paddle.nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool=True,
#         device=None, dtype=None, lora_r=8, lora_alpha=16, lora_dropout=0.05,
#         lora_len=0, **kwargs) ->None:
#         super().__init__(in_features=in_features, out_features=out_features, bias_attr=bias)
#         self.lora_r = lora_r
#         self.lora_alpha = lora_alpha
#         self.lora_len = lora_len
#         if lora_dropout > 0.0:
#             self.lora_dropout = paddle.nn.Dropout(p=lora_dropout)
#         else:
#             self.lora_dropout = lambda x: x
#         self.lora_scaling = self.lora_alpha / self.lora_r
#         self.Plora_A = paddle.nn.Linear(in_features=in_features,
#             out_features=self.lora_r, bias_attr=False)
#         self.Plora_B = paddle.nn.Linear(in_features=self.lora_r,
#             out_features=out_features, bias_attr=False)
#         self.reset_parameters()

#     def reset_parameters(self):
#         if hasattr(self, 'lora_A'):
#             init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
#                 negative_slope=math.sqrt(5), nonlinearity='leaky_relu')
#             init_KaimingUniform(self.lora_A.weight)
#             init_Constant = paddle.nn.initializer.Constant(value=0.0)
#             init_Constant(self.lora_B.weight)

#     def forward(self, x, im_mask=None):
#         res = super().forward(x)
#         if im_mask is not None:
#             if paddle.sum(x=im_mask) > 0:
#                 part_x = x[im_mask]
#                 res[im_mask] += self.Plora_B(self.Plora_A(self.lora_dropout
#                     (part_x))) * self.lora_scaling
#             else:
#                 part_x = x[:, :1]
#                 res[:, :1] += self.Plora_B(self.Plora_A(self.lora_dropout(
#                     part_x))) * 0
#         return res
