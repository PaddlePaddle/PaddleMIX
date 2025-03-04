import paddle

def all_gather(item):
    if paddle.distributed.is_initialized():
        global_item = []
        paddle.distributed.all_gather(
            global_item,std_grouped_rewards
        )
        return global_item
    else:
        return [item]