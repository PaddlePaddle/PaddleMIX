import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle import distributed as dist
import pprint

from .timm_ext import LabelSmoothingCrossEntropy
from paddlemix.models.common.distributed_utils import allgather


def gather_features_cat_group(image_features,
                              text_features,
                              group,
                              gather_with_grad=False):
    if group.world_size <= 1:
        return image_features, text_features
    features = paddle.concat([image_features, text_features], axis=-1)
    if gather_with_grad:
        features = allgather(features, group=group)
    else:
        gathered_features = []
        dist.all_gather(gathered_features, features, group=group)
        features = paddle.concat(gathered_features, axis=0)
    image_features, text_features = paddle.split(features, 2, axis=-1)
    return image_features, text_features


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
    shardinggroup = hcg.get_sharding_parallel_group()
    dpgroup = hcg.get_data_parallel_group()
    if gather_with_grad:
        if shardinggroup.nranks > 1:
            image_features, text_features = gather_features_cat_group(
                image_features, text_features, shardinggroup, gather_with_grad)
        if dpgroup.nranks > 1:
            image_features, text_features = gather_features_cat_group(
                image_features, text_features, dpgroup, gather_with_grad)
        all_image_features = image_features
        all_text_features = text_features
    else:
        image_features_bk = image_features
        text_features_bk = text_features
        if shardinggroup.nranks > 1:
            image_features, text_features = gather_features_cat_group(
                image_features, text_features, shardinggroup)
        if dpgroup.nranks > 1:
            image_features, text_features = gather_features_cat_group(
                image_features, text_features, dpgroup)
        if not local_loss:
            dp_rank = hcg.get_data_parallel_rank()
            sharding_rank = hcg.get_sharding_parallel_rank()
            sharding_size = hcg.get_sharding_parallel_world_size()
            loc = sharding_rank + sharding_size * dp_rank
            # ensure grads for local rank when all_* features don't have a gradient
            image_features = list(image_features.chunk(world_size, axis=0))
            text_features = list(text_features.chunk(world_size, axis=0))
            image_features[loc] = image_features_bk
            text_features[loc] = text_features_bk
            image_features = paddle.concat(image_features, axis=0)
            text_features = paddle.concat(text_features, axis=0)
        all_image_features = image_features
        all_text_features = text_features

    return all_image_features, all_text_features


def gather_features_bk(image_features,
                       text_features,
                       local_loss=False,
                       gather_with_grad=False,
                       rank=0,
                       world_size=1,
                       use_horovod=False):

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features_list = []
        allgather(all_image_features_list, image_features)
        all_image_features = paddle.concat(all_image_features_list, axis=0)
        all_text_features_list = []
        allgather(all_text_features_list, text_features)
        all_text_features = paddle.concat(all_text_features_list, axis=0)
    else:
        gathered_image_features = []
        gathered_text_features = []
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = paddle.concat(gathered_image_features, axis=0)
        all_text_features = paddle.concat(gathered_text_features, axis=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Layer):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1, ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        # self.prev_num_logits = 0
        # self.labels = {}

    def forward(self, preds):
        image_features, text_features, logit_scale = preds
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @all_text_features.T
                logits_per_text = logit_scale * text_features @all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @text_features.T
            logits_per_text = logit_scale * text_features @image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        # if self.prev_num_logits != num_logits or device not in self.labels:
        labels = paddle.arange(num_logits, dtype=paddle.int64)
        if self.world_size > 1 and self.local_loss:
            labels = labels + num_logits * self.rank
        #     if self.cache_labels:
        #         self.labels[device] = labels
        #         self.prev_num_logits = num_logits
        # else:
        #     labels = self.labels[device]
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss, logits_per_image, logits_per_text, labels


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1, ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size, )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = paddle.nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, preds, output_dict=False):
        image_features, text_features, logit_scale, logits, pred_labels = preds

        caption_loss = self.caption_loss(
            logits,
            pred_labels, )
        caption_loss = caption_loss * self.caption_loss_weight

        clip_loss = 0
        clip_loss, logits_per_image, logits_per_text, labels = super().forward(
            [image_features, text_features, logit_scale])
        clip_loss = self.clip_loss_weight * clip_loss

        if output_dict:  #output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss + caption_loss, logits_per_image, logits_per_text, labels


if __name__ == "__main__":
    pass
