import argparse


def get_default_params(model_name):
    model_name = model_name.lower()
    if 'vit' in model_name:
        return {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.98, 'epsilon': 1e-06}
    else:
        return {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08}


def get_args():
    parser = argparse.ArgumentParser(description='')

    # options for data config
    parser.add_argument(
        '--image_size', default=224, type=int, help='image size for training')
    parser.add_argument(
        '--vision_layers',
        default=12,
        type=int,
        help='# of ViT layers for training')
    parser.add_argument(
        '--vision_width',
        default=896,
        type=int,
        help='The width of ViT for training')
    parser.add_argument(
        '--vision_heads',
        default=14,
        type=int,
        help='The heads of ViT for training')
    parser.add_argument(
        '--transformer_width',
        default=768,
        type=int,
        help='The width of text-transformer for training')
    parser.add_argument(
        '--transformer_heads',
        default=12,
        type=int,
        help='# of head for text-transformer for training')
    parser.add_argument(
        '--transformer_layers',
        default=12,
        type=int,
        help='# of layers for text-transformer for training')
    parser.add_argument(
        '--vocab_size',
        default=50000,
        type=int,
        help='The vocab_size for tokens for training')
    parser.add_argument(
        '--embed_dim',
        default=256,
        type=int,
        help='The embedding dim for training')
    parser.add_argument(
        '-p',
        '--print_freq',
        default=10,
        type=int,
        metavar='N',
        help='print frequency (default: 10)')
    parser.add_argument(
        '-s',
        '--step_freq',
        default=2000,
        type=int,
        metavar='N',
        help='step frequency (default: 2000)')
    parser.add_argument(
        '--img_numbers',
        default=1000000,
        type=int,
        metavar='N',
        help='total image number')
    parser.add_argument(
        '--parts_dir',
        default='../xingtian_ROOT/dumeng/DATA/train_list_bin_2000.txt',
        type=str,
        help='hadoop parts dir')
    parser.add_argument(
        '--data_dir',
        default='../xingtian_ROOT/dumeng/DATA/fanli_cs_img_label_det_crop_right_b64_identity_after_aug_balance_v4',
        type=str,
        help='hadoop dataset dir with')
    parser.add_argument('--random_seed', type=int, default=2021, help='')

    # Training/Optimization parameters
    parser.add_argument(
        "--warmup",
        default=2000,
        type=int,
        help="Number of iters for the linear learning-rate warm up.")
    parser.add_argument(
        '--epochs',
        default=20,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.03,
        type=float,
        metavar='LR',
        help='initial learning rate',
        dest='lr')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument(
        '--wd',
        '--weight_decay',
        default=0.2,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay')
    parser.add_argument(
        '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument(
        '-b',
        '--batch_size',
        default=128,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument(
        '--SAVE_PATH',
        default="./saved_model",
        type=str,
        help='path to save model checkpoint')
    parser.add_argument(
        '--SAVE_PATH_STEP',
        default="./saved_model_step",
        type=str,
        help='path to save model checkpoint for step')

    # options for distribute training
    parser.add_argument(
        '--dist_url',
        default='127.0.0.1',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--seed', default=0, type=int, help='seed for initializing training.')
    parser.add_argument(
        '--ngpus_per_node',
        default=4,
        type=int,
        help='number of GPUs in each node/machine.')

    parser.add_argument('--context_length', type=int, default=76, help="")
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument(
        '--use_amp',
        type=str,
        default="O1",
        help='whether use amp, should be one of [O1, O2, OFF]')
    parser.add_argument(
        '--half_set',
        type=str,
        default="float16",
        help='which half precision to use, should be one of [float16, bfloat16]'
    )
    parser.add_argument(
        '--optimizer', type=str, default="lamb", help='optimizer setting')
    parser.add_argument("--model", type=str, default=None, help="model name.")
    parser.add_argument(
        "--text-lr",
        type=float,
        default=None,
        help="Learning rate of text encoder.")
    parser.add_argument(
        "--visual-lr",
        type=float,
        default=None,
        help="Learning rate of visual encoder.")

    parser.add_argument(
        "--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument(
        "--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument(
        "--epsilon", type=float, default=None, help="Adam epsilon.")

    parser.add_argument(
        "--text-wd",
        type=float,
        default=None,
        help="Weight decay of text encoder.")
    parser.add_argument(
        "--visual-wd",
        type=float,
        default=None,
        help="Weight decay of visual encoder.")

    parser.add_argument(
        "--ld", type=float, default=1.0, help="Learning rate Layer decay.")
    parser.add_argument(
        "--text-ld",
        type=float,
        default=None,
        help="Learning rate Layer decay of text encoder.")
    parser.add_argument(
        "--visual-ld",
        type=float,
        default=None,
        help="Learning rate Layer decay of visual encoder.")

    # coca
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa.")
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa.")

    parser.add_argument(
        "--log_local",
        default=False,
        action="store_true",
        help="Whether to show log only on local.")
    parser.add_argument(
        "--profiler",
        default=False,
        action="store_true",
        help="Whether to show log only on local.")
    parser.add_argument(
        '--patch_size',
        default=16,
        type=int,
        metavar='N',
        help='vit patch size')
    parser.add_argument(
        '--paddle_model',
        type=str,
        default='./paddle_model',
        help='The path of paddle model')
    parser.add_argument(
        '--model_name',
        type=str,
        default='default',
        help='The task name of paddle model')
    parser.add_argument(
        '--gmcnt', type=int, default=1, help="gradient merge steps")
    parser.add_argument(
        '-af',
        '--accum_freq',
        type=int,
        default=1,
        metavar='N',
        help='accum frequency (default: 1)')
    parser.add_argument(
        '--dp_degree', type=int, default=2, help="data parallel degrees")
    parser.add_argument(
        '--mp_degree', type=int, default=1, help="model parallel degrees")
    parser.add_argument(
        '--pp_degree', type=int, default=1, help="pipeline parallel degrees")
    parser.add_argument(
        '--sharding_degree',
        type=int,
        default=4,
        help="sharding parallel degrees")
    parser.add_argument(
        '--last_epoch', type=int, default=-1, help="the last epoch to resume")
    args = parser.parse_args()
    default_params = get_default_params("EVA-CLIP")
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    return args
