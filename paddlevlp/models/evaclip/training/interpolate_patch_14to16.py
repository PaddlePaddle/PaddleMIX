import sys
import paddle
import argparse


def interpolate_pos_embed(checkpoint_model, new_size=16):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = 196
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
        checkpoint_model['pos_embed'] = new_pos_embed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='interpolate patch_embed kernel')
    parser.add_argument(
        '--input',
        default='/path/to/eva_psz14.pt',
        type=str,
        metavar='PATH',
        required=True,
        help='path to input EVA checkpoint with patch_embed kernel_size=14x14')
    parser.add_argument(
        '--output',
        default='/path/to/eva_psz14to16.pt',
        type=str,
        metavar='PATH',
        required=True,
        help='path to output EVA checkpoint with patch_embed kernel_size=16x16')
    args = parser.parse_args()
    checkpoint = paddle.load(args.input)
    patch_embed = checkpoint['patch_embed.proj.weight']
    C_o, C_in, H, W = patch_embed.shape
    patch_embed = paddle.nn.functional.interpolate(
        x=patch_embed.astype(dtype='float32'),
        size=(16, 16),
        mode='bicubic',
        align_corners=False)
    checkpoint['patch_embed.proj.weight'] = patch_embed
    interpolate_pos_embed(checkpoint, new_size=16)
    print('======== new state_dict ========')
    for k, v in list(checkpoint.items()):
        print(k, '        ', v.shape)
    paddle.save(obj=checkpoint, path=args.output, protocol=4)
