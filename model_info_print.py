import argparse
from utils.param_FLOPs_counter import model_info
from utils.Logger import Logger


from ssd.model import SSD300,ResNet
parser = argparse.ArgumentParser(description='Train Single Shot MultiBox Detector on COCO')
parser.add_argument('--model_name', default='SSD300', type=str,
                    help='The model name')

parser.add_argument('--backbone', type=str, default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--backbone-path', type=str, default=None,
                    help='Path to chekcpointed backbone. It should match the'
                            ' backbone model declared with the --backbone argument.'
                            ' When it is not provided, pretrained model from torchvision'
                            ' will be downloaded.')

args = parser.parse_args()
model = ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))

log = Logger('logs/'+ args.model_name+'_INFO.log',level='debug')
model_info(model,log,img_size=[300,300])

