import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR

from data.coco import COCODetection
from data.utils.data_aug import SSDTransformer
from data.utils.utils import DefaultBoxes, Encoder
from ssd.evaluate import evaluate
from ssd.model import SSD300, Loss, ResNet
from ssd.train_utils import load_checkpoint, tencent_trick, train_loop
from utils.Logger import Logger
from utils.multi_gpu import init_distributed_mode


# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
# https://catalog.ngc.nvidia.com/models
def main():
    parser = argparse.ArgumentParser(description='Train Single Shot MultiBox Detector on COCO')
    parser.add_argument('--model_name', default='SSD300', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/SSD300.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='data/coco.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='3,4', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--checkpoint', default=None, help='The checkpoint path')
    parser.add_argument('--save', type=str, default='checkpoints',
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of epochs for training') #default 65
    # parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
    #                     help='epochs at which to evaluate')
    #TODO need less
    parser.add_argument('--evaluation', nargs='*', type=int, default=[1000],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--seed', '-s', default = 42 , type=int, help='manually set random seed for torch')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=2.6e-3,
                        help='learning rate for SGD optimizer')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0.0005,
                        help='weight-decay for SGD optimizer')
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
    parser.add_argument('--report-period', type=int, default=100, help='Report the loss every X times.')
    
    # parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    
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
    #Random seed
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    

    
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
    args.lr = args.lr * args.NUM_gpu * (args.batch_size / 32)
    #Logger
    log_path = '{}-{}-lr-{}-{}'.format(args.model_name, data_cfg.NAME, args.lr, time.strftime('%Y%m%d-%H'))
    
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
 
    train_dataset = COCODetection(root=args.data.DATASET_PATH,image_set='train2017', 
                        transform=SSDTransformer(dboxes))

    val_dataset = COCODetection(root=args.data.DATASET_PATH,image_set='val2017', 
                        transform=SSDTransformer(dboxes, val=True))
    
    if args.multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=train_shuffle, 
                                  sampler=train_sampler,
                                  pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)
    cocoGt = val_dataset.coco
    
    inv_map = {v: k for k, v in val_dataset.label_map.items()}   # label map  90 ids -> 80 classes
    # Load model
    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))

    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)
    
    ssd300 = ssd300.cuda()
    loss_func = loss_func.cuda()   
    
    if args.multi_gpu:
        # DistributedDataParallel
        ssd300 = DDP(ssd300, device_ids=[args.local_rank], output_device=args.local_rank)
        
        
    optimizer = torch.optim.SGD(params=tencent_trick(ssd300), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
 

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.multi_gpu else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    total_time = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # Automatic Mixed Precision
    for epoch in range(start_epoch, args.epochs):
        
        if args.multi_gpu :
            train_loader.sampler.set_epoch(epoch)
        start_epoch_time = time.time()
        iteration = train_loop(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, iteration,
                                    args, log)
        scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            log.logger.info('Epoch:',epoch,'Use Time:', end_epoch_time) 
            
        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_loader, cocoGt, encoder, inv_map, args, log)

            if args.local_rank == 0:
                log.logger.info('Epoch:',epoch,'Acc:', acc)
        # Save model
        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict()}
            if args.multi_gpu:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            log.logger.info('model path:', save_path)
        if args.local_rank == 0:
            log.logger.info('total time:', total_time )


   
    # #Save model   
        
    #     test_accuracy = compute_accuracy(args,test_dataloader,net)
    #     log.logger.info('test_accuracy is: %s',test_accuracy)
    #     scheduler.step(mean_loss)
        
        
    #     if  test_accuracy > best_test_accuracy :
    #         best_epoch = epoch
    #         best_test_accuracy = test_accuracy
    #         print('Best acc is:',best_test_accuracy)
    #         save_path = args.CHECKPOINT_DIR+'/'+args.model_name+'.pth'
    #         torch.save(net.state_dict(), save_path)
    #     if args.save_data == True:
    #         save_loss.append(mean_loss.item())
    #         save_acc.append(test_accuracy)
    # log.logger.info('Best acc is: %s \n,Best epoch is: %s',best_test_accuracy,best_epoch)
    # if args.save_data == True:
    #     #字典中的key值即为csv中列名
    #     dataframe = pd.DataFrame({'Epoch_loss':save_loss,'val_acc':save_acc})
    #     #将DataFrame存储为csv,index表示是否显示行名，default=True
    #     dataframe.to_csv('output/plot_data/'+args.model_name+'.csv',index=False,sep=',')



if __name__ == '__main__':
    main()
