import os
import yaml
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader

from data_DALI_useless.utils.utils import DefaultBoxes, COCODetection
from data_DALI_useless.utils.utils import SSDTransformer
from pycocotools.coco import COCO
#DALI import
from data_DALI_useless.coco_pipeline import COCOPipeline, DALICOCOIterator

def get_train_loader(args, local_seed):
    train_annotate = os.path.join(args.data.DATASET_PATH, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data.DATASET_PATH, "images/train2017")



    
    train_pipe = COCOPipeline(batch_size=args.batch_size,
        file_root=train_coco_root,
        annotations_file=train_annotate,
        default_boxes=DefaultBoxes(args.model.figsize, args.model.feat_size, args.model.steps, args.model.scales, args.model.aspect_ratios),
        device_id=args.local_rank,
        num_shards=args.NUM_gpu,
        output_fp16=args.amp,
        output_nhwc=False,
        pad_output=False,
        num_threads=args.num_workers, seed=local_seed)
    train_pipe.build()
    test_run = train_pipe.schedule_run(), train_pipe.share_outputs(), train_pipe.release_outputs()
    train_loader = DALICOCOIterator(train_pipe, 118287 / args.NUM_gpu)
    return train_loader


def get_val_dataset(args):
    dboxes = DefaultBoxes(args.model.figsize, args.model.feat_size, args.model.steps, args.model.scales, args.model.aspect_ratios)
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.data.DATASET_PATH, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data.DATASET_PATH, "images/val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def get_val_dataloader(dataset, args):
    if args.distributed:
        val_sampler = torch.utils.data_DALI_useless.distributed.DistributedSampler(dataset)
    else:
        val_sampler = None

    val_dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)

    return val_dataloader

def get_coco_ground_truth(args):
    val_annotate = os.path.join(args.data.DATASET_PATH, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt