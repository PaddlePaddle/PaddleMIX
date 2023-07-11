import paddle
import os
import json


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if paddle.distributed.is_initialized():
        if paddle.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_dist_avail_and_initialized():
    if not paddle.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return paddle.distributed.get_world_size()


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID',
              'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS',
              'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def init_distributed_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    if is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            # paddle.distributed.init_process_group(backend=args.dist_backend,
            #     init_method=args.dist_url, world_size=args.world_size, rank
            #     =args.rank)
            paddle.distributed.init_parallel_env()
        else:
            args.local_rank, _, _ = world_info_from_env()
            # paddle.distributed.init_process_group(backend=args.dist_backend,
            #     init_method=args.dist_url)
            paddle.distributed.init_parallel_env()
            args.world_size = paddle.distributed.get_world_size()
            args.rank = paddle.distributed.get_rank()
        args.distributed = True
    if paddle.device.cuda.device_count() >= 1:
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        paddle.device.set_device(device=device)
    else:
        device = 'cpu'
    args.device = device
    return device
