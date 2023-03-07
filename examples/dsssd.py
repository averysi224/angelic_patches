import os
import torch
import torch.utils.data
import torchvision
from PIL import Image, ImageChops
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage.transform import resize


from matplotlib.patches import Polygon

import numpy as np

import pdb

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return False
    return True

class SSDDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.mask_transforms = get_mask_transform()

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        if is_greyscale(img):
            img = Image.new("RGB", img.size)
        # print(img)
        # print(img.size)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        width, height = img.size
        boxes = []
        masks = []
        # background 0; object 1
        # mask_combo = np.zeros([height, width], dtype=np.uint8)
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            
            normalized_x = int(xmin * 300 / width)
            normalized_y = int(ymin * 300 / height)
            normalized_width = int((xmax - xmin) * 300 / width)
            normalized_height = int((ymax - ymin) * 300 / height)
            boxes.append([normalized_x, 
                          normalized_y, 
                          normalized_x+normalized_width, 
                          normalized_y+normalized_height])
            labels[i] = coco_annotation[i]['category_id']

            ann = coco_annotation[i]
            if 'segmentation' in ann:
                mask = coco.annToMask(ann)
                maskk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
                maskk = self.mask_transforms(maskk)
                masks.append(maskk)
        # print(len(masks), masks[0].shape, masks[-1].shape)
        # plt.imshow(masks[0].T)
        # plt.savefig("testtest0.5/mask{}.png".format(index))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # masks = torch.as_tensor(masks, dtype=torch.float32)
        
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        # it seems like iscrowd is always 0
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            # float32
            img = self.transforms(img)
            # print(img.shape, img.max(), img.min())
            # plt.imshow(img.T)
            # plt.savefig("testtest0.5/image{}.png".format(index))

        # may stack masks
        return img, my_annotation, masks

    def __len__(self):
        return len(self.ids)

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize([300, 300]))
    custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)

    
def get_mask_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToPILImage())
    custom_transforms.append(torchvision.transforms.Resize([300, 300]))
    custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)
