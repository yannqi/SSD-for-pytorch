import os
import argparse
import yaml
import tqdm
import time
import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from utils.Logger import Logger
from utils.utils import generate_mean_std

from ssd.model import SSD300, ResNet, Loss
from ssd.train_utils import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
def main():
    parser = argparse.ArgumentParser(description='Train Single Shot MultiBox Detector on COCO')
    parser.add_argument('--model_name', default='SSD300', type=str,
                        help='The model name')
    parser.add_argument('--model_config', default='configs/SSD300.yaml', 
                        metavar='FILE', help='path to model cfg file', type=str,)
    parser.add_argument('--data_config', default='data/coco.yaml', 
                        metavar='FILE', help='path to data cfg file', type=str,)
    parser.add_argument('--device_gpu', default='0,1,3', type=str,
                        help='Cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--checkpoint', default=None, help='The checkpoint path')
    parser.add_argument('--save', type=str, default=None,
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--epochs', '-e', type=int, default=65,
                        help='number of epochs for training')
    parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
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
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='weight-decay for SGD optimizer')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    # parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    
    # Multi Gpu
    parser.add_argument('--multi_gpu', default=True, type=bool,
                        help='Whether to use multi gpu to train the model')
    
    #others 
    parser.add_argument('--amp', action='store_true', default = True,
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')

    args = parser.parse_args()
    
    # Pre load 
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank == 0:
        os.makedirs('./local_rank', exist_ok=True)
    # write json only on the main thread

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        args.epochs = 1
    else:
        train_loop_func = train_loop
    
    
    
    
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
    
    # Use Gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    args.device = device
    
    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.NUM_gpu = torch.distributed.get_world_size()
    else:
        args.NUM_gpu = 1
    
    #Logger
    log_path = '{}-{}-lr-{}-{}'.format(args.model_name, data_cfg.NAME, args.lr, time.strftime('%Y%m%d-%H'))
    
    log = Logger('logs/'+log_path+'.log',level='debug')

    #Initial Logging
    log.logger.info('gpu device = %s' % args.device_gpu)
    log.logger.info('args = %s', args)
    log.logger.info('data_cfgs = %s', data_cfg)
    
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    
    # Pre dataset
    
    cocoGt = get_coco_ground_truth(args)
    # Use DALICOCO 
    train_loader = get_train_loader(args, args.seed - 2**31)
    # Use Traditional coco 
    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    # Setup data, defaults
    dboxes = DefaultBoxes(args.model.figsize, args.model.feat_size, 
                          args.model.steps, args.model.scales, args.model.aspect_ratios)
    encoder = Encoder(dboxes)
    # Load model
    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))
    #The learning rate is automatically scaled 
    # (in other words, multiplied by the number of GPUs and multiplied by the batch size divided by 32).
    args.lr = args.lr * args.NUM_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)
    
 
    if args.device == 'cuda':
        ssd300.cuda()
        loss_func.cuda()   
    
    
    
    optimizer = torch.optim.SGD(params=tencent_trick(ssd300), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
 
    # if args.distributed:
    #     ssd300 = DDP(ssd300) #TODO understand the DDP 

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}   # label map  90 ids -> 80 classes

    total_time = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # Automatic Mixed Precision
    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    Logger, args, mean, std)
        if args.mode in ["training", "benchmark-training"]:
            scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            logger.log('model path', save_path)
        train_loader.reset()
    DLLogger.log((), { 'total time': total_time })
    logger.log_summary()

   
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