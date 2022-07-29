#exmaple: 1 node,  2 GPUs per node (2GPUs)

CUDA_VISIBLE_DEVICES=3,4 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    train.py --multi_gpu=True

# CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=localhost \
#     --master_port=22222 \
#     train.py





# https://zhuanlan.zhihu.com/p/360405558
# https://juejin.cn/post/7044336367588868109