#YannQi
import os 
import sys 
import torch
import yaml
import numpy as np
import torch.utils.data as data
from PIL import Image
COCO_ROOT ='data'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    """Get the label map from label name to index."""
    label_map = {}
    labels = open(label_file,'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform():
    """
    Transforms a COCO annotation into a Tensor of bbox coords and label index.
    Initilized with a dictionary lookup of classnames to indexes.
    """
    def __init__(self):
        self.label_map = get_label_map(os.path.join(COCO_ROOT, 'coco_labels.txt'))
    def __call__(self, target, width, height):
        """ Transforms a COCO annotation into a Tensor of bbox coords and label index.

        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)  #ltrb styel(left top right bottom)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx] ltrb
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]  

class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train2017', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        self.label_map = get_label_map(os.path.join(COCO_ROOT, 'coco_labels.txt'))
        sys.path.append(os.path.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = os.path.join(root, IMAGES, image_set)
        self.coco = COCO(os.path.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys()) #connect image with annotations
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        if image_set == 'train2017':
            self.type = 'train'
        else : self.type = 'test'
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        img, boxes, labels, height, width, img_id = self.pull_item(index)
    
        return img, img_id, (height, width), boxes, labels   # gt : [xmin, ymin, xmax, ymax, label_idx]

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        # target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        #img = cv2.imread(os.path.join(self.root, path))
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        width, height = img.size

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
            target = torch.tensor(target, dtype=torch.float)
            boxes = target[:, :4].float()
            labels = target[:, 4].long() +1 #Note: Add one to make zero as background. 
        if self.transform is not None:    
            img, (height, width), boxes, labels = \
                self.transform(img, (height, width), boxes, labels)
                
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, boxes, labels, height, width, img_id
        #target : [[xmin, ymin, xmax, ymax(ltrb), label_idx], ... ]
    # def pull_image(self, index):
    #     '''Returns the original image object at index in PIL form

    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.

    #     Argument:
    #         index (int): index of img to show
    #     Return:
    #         cv2 img
    #     '''
    #     img_id = self.ids[index]
    #     path = self.coco.loadImgs(img_id)[0]['file_name']
    #     return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return img_id,self.coco.loadAnns(ann_ids)

    def __repr__(self):
        #return COCODetection() -> return COCODetection().__repr__
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def collate_fn(batch):
        """If in SSD model, it is useless."""
        images, targets = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        
        boxes = []
        labels = []
        img_id = []
        for t in targets:
            boxes.append(t['boxes'])
            labels.append(t['labels'])
            img_id.append(t["image_id"])
        targets = {"boxes": torch.stack(boxes, dim=0),
                   "labels": torch.stack(labels, dim=0),
                   "image_id": torch.as_tensor(img_id)}
        return images, targets
if __name__ == "__main__":
    
    #Load config
    cfg_path = open('configs/COCO_config.yaml')
    # 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
    from easydict import EasyDict as edict
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg) # 将普通的字典传入到edict()
    
    COCO_ROOT = os.path.join(cfg.DATASET.DATASET_PATH)
    #print(get_label_map(os.path.join(COCO_ROOT, 'coco_labels.txt')))
    dataset = COCODetection(root=COCO_ROOT,transform=None)
    print(dataset)
    print(len(dataset))
    print(dataset[5])
    