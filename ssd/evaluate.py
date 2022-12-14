import io
import time
from contextlib import redirect_stdout

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval


def evaluate(model, coco, cocoGt, encoder, inv_map, args, log):
    if args.multi_gpu:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    
    
    ret = []
    start = time.time()

    # for idx, image_id in enumerate(coco.img_keys):
    for nbatch, (img, img_id, height, width, _, label) in enumerate(coco):
        with torch.no_grad():
            img = img.cuda()

        
            with torch.cuda.amp.autocast(enabled=args.amp):
                # Get predictions
                ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            N = ploc.shape[0]  # Batch_size
            for idx in range(N):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue
    
                
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0] * width[idx], \
                                loc_[1] * height[idx],
                                (loc_[2] - loc_[0]) * width[idx],
                                (loc_[3] - loc_[1]) * height[idx],
                                prob_,
                                inv_map[label_]])


    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if args.multi_gpu:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()
        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]

        torch.cuda.synchronize()
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
        torch.cuda.synchronize()

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results
 
        torch.cuda.synchronize()  # ?????????????????????????????????????????????????????????
        torch.distributed.all_gather(other_ret, ret_pad)
        torch.cuda.synchronize()

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
        # Otherwise full results are just our results
        final_results = ret

    if args.local_rank == 0:
         log.logger.info("Predicting Ended, total time: {:.2f} s".format(time.time() - start))
    # Save result as json
    # import json
    # file_name = 'number.json' #?????????????????????????????????????????????json??????
    # with open(file_name,'w') as file_object:
    #     json.dump(final_results.tolist(),file_object)
    cocoDt = cocoGt.loadRes(final_results)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    if args.local_rank == 0:
        E.summarize()
        for i in E.stats:
            log.logger.info("Current AP: {:.5f}".format(i))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E.summarize()

    # put your model in training mode back on
    model.train()

    return E.stats[0]  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
