import os
import torch
import torch.distributed as dist
def init_distributed_mode(args):
    # set up distributed device
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)
    print(args.device,'argsdevice')
    args.NUM_gpu = torch.distributed.get_world_size()
    print(f"[init] == local rank: {args.local_rank}, global rank: {args.rank} ==")