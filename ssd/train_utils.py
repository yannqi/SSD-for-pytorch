from torch.autograd import Variable
import torch
import time

from apex import amp




def train_loop(model, loss_func, scaler, epoch, optim, train_dataloader, iteration, args,log):
    """Traditional dataloader style."""
    for nbatch, (img, _, (_, _), bbox, label) in enumerate(train_dataloader):
        
        img = img.cuda()
        bbox = bbox.cuda() # [N, 8732, 4] N is batch_size
        label = label.cuda() # [N, 8732]  value 0: backgruond iou<0.5
            
    
        with torch.cuda.amp.autocast(enabled=args.amp):  # Automatic Mixed Precision
            ploc, plabel = model(img)

            ploc, plabel = ploc.float(), plabel.float()
            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
            gloc = Variable(trans_bbox, requires_grad=False)  # [batch_size, 4, 8732]
            glabel = Variable(label, requires_grad=False) # [batch_siz]

            loss = loss_func(ploc, plabel, gloc, glabel)

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)
            
         # scaler Automatic Mixed Precision
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        if args.local_rank == 0 and iteration % args.report_period == 0:   
            log.logger.info('Epoch: %s, iteration: %s , loss: %s',epoch, iteration, loss.item())
        iteration += 1

    return iteration

def train_loop_DALI(model, loss_func, scaler, epoch, optim, train_dataloader, iteration, logger, args, mean, std):
    for nbatch, data in enumerate(train_dataloader):
        img = data[0][0][0] # [batch_size, 3, 300, 300]
        bbox = data[0][1][0]
        label = data[0][2][0] 
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)
        if args.device == 'cuda':
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue

        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N  #Total 8732 default box. 
        bbox = bbox.view(N, M, 4) # [batch_size, 8732, 4]
        label = label.view(N, M) # [batch_size, 8732]    value 0 : background

        with torch.cuda.amp.autocast(enabled=args.amp):  # Automatic Mixed Precision
            ploc, plabel = model(img)

            ploc, plabel = ploc.float(), plabel.float()
            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
            gloc = Variable(trans_bbox, requires_grad=False)  # [batch_size, 4, 8732]
            glabel = Variable(label, requires_grad=False) # [batch_siz]

            loss = loss_func(ploc, plabel, gloc, glabel)

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        # scaler Automatic Mixed Precision
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())
        iteration += 1

    return iteration

def benchmark_train_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    start_time = None
    # tensor for results
    result = torch.zeros((1,)).cuda()
    for nbatch, data in enumerate(loop(train_dataloader)):
        if nbatch >= args.benchmark_warmup:
            torch.cuda.synchronize()
            start_time = time.time()

        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)
        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue

        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        with torch.cuda.amp.autocast(enabled=args.amp):
            ploc, plabel = model(img)

            ploc, plabel = ploc.float(), plabel.float()
            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
            gloc = Variable(trans_bbox, requires_grad=False)
            glabel = Variable(label, requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if nbatch >= args.benchmark_warmup + args.benchmark_iterations:
            break

        if nbatch >= args.benchmark_warmup:
            torch.cuda.synchronize()
            logger.update(args.batch_size*args.N_gpu, time.time() - start_time)

    result.data[0] = logger.print_result()
    if args.N_gpu > 1:
        torch.distributed.reduce(result, 0)
    if args.local_rank == 0:
        print('Training performance = {} FPS'.format(float(result.data[0])))


def loop(dataloader, reset=True):
    while True:
        for data in dataloader:
            yield data
        if reset:
            dataloader.reset()

def benchmark_inference_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    model.eval()
    val_datas = loop(val_dataloader, False)

    for i in range(args.benchmark_warmup + args.benchmark_iterations):
        torch.cuda.synchronize()
        start_time = time.time()

        data = next(val_datas)
        img = data[0]
        with torch.no_grad():
            if not args.no_cuda:
                img = img.cuda()
            img.sub_(mean).div_(std)
            with torch.cuda.amp.autocast(enabled=args.amp):
                _ = model(img)

        torch.cuda.synchronize()
        end_time = time.time()


        if i >= args.benchmark_warmup:
            logger.update(args.eval_batch_size, end_time - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """Load model from checkpoint."""
    
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
    
    
    