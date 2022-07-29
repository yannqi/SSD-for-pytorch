import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP


from data.coco import COCODetection
from data.utils.data_aug import SSDTransformer
from data.utils.utils import DefaultBoxes, Encoder
from ssd.evaluate import evaluate
from ssd.model import SSD300, ResNet
from ssd.train_utils import load_checkpoint
from utils.Logger import Logger
from utils.multi_gpu import init_distributed_mode


# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
def main():
    parser = argparse.ArgumentParser(description='Train Single Shot MultiBox Detector on COCO')
    parser.add_argument('--model_name', default='SSD300', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/SSD300.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='data/coco.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='1', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--checkpoint', default='nvidia_ssdpyt_fp16_190826.pt', help='The checkpoint path')


    # Hyperparameters

    parser.add_argument('--batch_size', '--bs', type=int, default=64,
                        help='number of examples for each iteration')
    parser.add_argument('--num_workers', type=int, default=8) 
    
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
 
    # Multi Gpu
    parser.add_argument('--multi_gpu', default=False, type=bool,
                        help='Whether to use multi gpu to train the model, if use multi gpu, please use by sh.')
    
    #others 
    parser.add_argument('--amp', action='store_true', default = False,
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')

    args = parser.parse_args()
    
    
    data_cfg_path = open(args.data_config)
    # 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
    from easydict import EasyDict as edict
    data_cfg = yaml.full_load(data_cfg_path)
    data_cfg = edict(data_cfg) 
    args.data = data_cfg
    
    cfg_path = open(args.model_config)
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg) 
    args.model = cfg

    
    # Initialize Multi GPU 

    if args.multi_gpu == True :
        init_distributed_mode(args)
    else: 
        # Use Single Gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')
        args.device = device   
        args.NUM_gpu = 1
        args.local_rank = 0

    
    #The learning rate is automatically scaled 
    # (in other words, multiplied by the number of GPUs and multiplied by the batch size divided by 32).
    #Logger
    log_path = 'Test-{}-lr-{}-{}'.format(args.model_name, data_cfg.NAME, time.strftime('%Y%m%d-%H'))
    
    log = Logger('logs/'+log_path+'.log',level='debug')

    #Initial Logging
    if args.local_rank == 0:
        log.logger.info('gpu device = %s' % args.device_gpu)
        log.logger.info('args = %s', args)
        log.logger.info('data_cfgs = %s', data_cfg)
    
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    
    # Pre dataset       dboxes : 8732 default box
    dboxes = DefaultBoxes(args.model.figsize, args.model.feat_size,     
                args.model.steps, args.model.scales, args.model.aspect_ratios) 

    encoder = Encoder(dboxes)
 
    val_dataset = COCODetection(root=args.data.DATASET_PATH,image_set='val2017', 
                        transform=SSDTransformer(dboxes, val=True))
    
    if args.multi_gpu:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)     
    else:

        val_sampler = None


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)
    cocoGt = val_dataset.coco
    inv_map = {v: k for k, v in val_dataset.label_map.items()}   # label map  90 ids -> 80 classes
    # Load model
    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))

    ssd300 = ssd300.cuda()
    
    if args.multi_gpu:
        # DistributedDataParallel
        ssd300 = DDP(ssd300, device_ids=[args.local_rank], output_device=args.local_rank)
        
        

 

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.multi_gpu else ssd300, args.checkpoint)

        else:
            print('Provided checkpoint is not path to a file')
            return

    
    acc = evaluate(ssd300, val_loader, cocoGt, encoder, inv_map, args, log)



if __name__ == '__main__':
    main()
