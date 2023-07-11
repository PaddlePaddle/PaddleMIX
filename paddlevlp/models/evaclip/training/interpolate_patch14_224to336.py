import paddle
from math import pi


def interpolate_pos_embed(checkpoint):
    state_dict = paddle.load(checkpoint)
    checkpoint_model = state_dict['module']
    if 'visual.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['visual.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (336 // 14)**2
        num_extra_tokens = 1
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        new_size = int(num_patches**0.5)
        if orig_size != new_size:
            print('Position interpolate from %dx%d to %dx%d' %
                  (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            (-1, orig_size, orig_size,
             embedding_size)).transpose(perm=[0, 3, 1, 2])
        pos_tokens = paddle.nn.functional.interpolate(
            x=pos_tokens,
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False)
        pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(
            start_axis=1, stop_axis=2)
        new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
        checkpoint_model['visual.pos_embed'] = new_pos_embed
        state_dict['module'] = checkpoint_model
        paddle.save(
            obj=state_dict,
            path=checkpoint.replace('states', 'states_interp'),
            protocol=4)


if __name__ == '__main__':
    interpolate_pos_embed('/path/to/model_psz_14_224_ckpt')
