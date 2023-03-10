import torch
import torchvision
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchFasterRCNN
from art.estimators.object_detection import PyTorchSSD
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DPatch
from pycocotools import mask as maskUtils

import pdb
import os
import json
import copy

import dsld

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename


os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')

cate_list = {"person":1, "bus":6, "bottle": 44, "bowl":51, "chair":62, "laptop":73}
# Batch size
train_batch_size = 1
length = 224

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        #return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def extract_predictions(predictions_, name, cls, thresh=0.5, eps=1):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]

    # Get the predicted bounding boxes
    predictions_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(predictions_["boxes"].astype(int))]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])

    # Get a list of index with score greater than threshold
    threshold = thresh

    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
    if len(predictions_t) == 0:
        return

    predictions_t = predictions_t[-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]
    predictions_score = predictions_score[: predictions_t + 1]
    predictions_["labels"] = predictions_["labels"][: predictions_t + 1]

    ## Count how many objects' score is higher than threshold
    tgt = (predictions_["labels"] == cls)
    tgt_idx = []
    for ii in range(len(tgt)):
        if tgt[ii]:
            tgt_idx.append(ii)
    count = predictions_["labels"][tgt].shape[0]

    return predictions_["labels"], predictions_boxes, predictions_score, count

def plot_image_with_boxes(img, boxes, pred_cls, cls, gt_boxes=None):
    text_size = 1
    text_th = 2
    rect_th = 2
    for i in range(len(boxes)):
        if pred_cls[i] == cls:
            # Draw Rectangle with the coordinates, green
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=1)
            # Write the prediction class
            # cv2.putText(img, str(pred_cls[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), thickness=text_th)
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            gbox = [(gt_boxes[i][0], gt_boxes[i][1]), (gt_boxes[i][2], gt_boxes[i][3])]
            # Draw Rectangle with the coordinates
            cv2.rectangle(img, gbox[0], gbox[1], color=(0, 255, 255), thickness=1)

    plt.axis("off")
    img = img[:, :, ::-1]
    
    plt.imshow(img.astype(np.uint8), interpolation="nearest")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cate', default="person", help = "target test category")
    parser.add_argument('--clear', action="store_true", help = "test without corruption, default with corruption.")
    parser.add_argument('--model_name', default="frcnn", help = "choose from frcnn and ssd.")
    parser.add_argument('--coco_path', default='/data5/wenwens/coco2017/train2017', help = "path of COCO dataset")
    args = parser.parse_args()

    cate = args.cate # category name
    test_clear = args.clear
    CLS=cate_list[cate]  # category label
    model_name = args.model_name

    DIR_BASE = 'results/{}/affine_{}'.format(model_name, cate) 
    im_length = 224 if model_name == "frcnn" else 300
    path2data =  args.coco_path # path to coco dataset
    path2json = '../coco-manager/instances_'+cate+'_train2017.json'  # filtered single category json

    if model_name == "frcnn":
        mdl = PyTorchFasterRCNN(
            clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
        )
        # create own Dataset
        original_loss_history = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}
    else:
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        model.eval()
        mdl = PyTorchSSD(
            clip_values=(0, 255), model=model, attack_losses=["bbox_regression", "classification"]
        )
        original_loss_history = {"bbox_regression": 0, "classification": 0}

    my_dataset = dsld.myOwnDataset(root=path2data, annotation=path2json, im_length=im_length)
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    # select device (whether GPU or CPU)
    image = []
    gts_boxes = []
    gts_labels = []
    img_ids = []
    all_masks = []

    prefix = "Clear" if test_clear else "Frost"
    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations, masks] in enumerate(data_loader):
        if len(image) >=200:    # if too much data, use first 4000
            break
        if idx % 1000 == 0:
            print(idx)
        
        imgs = torch.stack(imgs)
        imgs = imgs.permute(0,2,3,1)
        imgs = imgs.numpy()[:, :,:, ::-1]*255.0
        if np.max(imgs) == 0:
            continue
        for i in range(imgs.shape[0]):
            # Process predictions
            try:
                # Plot predictions
                gt_class, gt_boxes = annotations[i]['labels'].numpy(), annotations[i]['boxes'].numpy().astype(int)
                count = np.sum(gt_class == 1)
                flag = False
                for g in range(len(gt_boxes)):
                    box = gt_boxes[g] 
                    if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                        flag = True
                if count > 0 and not flag:
                    gts_boxes.append(gt_boxes)
                    gts_labels.append(gt_class*CLS)
                    img_ids.append(annotations[i]["image_id"])
                    image.append(imgs[i])
                    all_masks.append(masks[0])

            except Exception as e:
                pass
    
    ratio = 0.1
    image = np.stack(image)
    trainSize = int(image.shape[0] * ratio)
    print(image.shape, "all", idx)
    test_images = image[trainSize:,]
    test_images = np.transpose(test_images, (0,3,1,2))

    test_gts_boxes = gts_boxes[trainSize:]
    test_gts_labels = gts_labels[trainSize:]
    test_ids = img_ids[trainSize:]
    # train images
    image = image[:trainSize,]
    gts_boxes = gts_boxes[:trainSize]
    gts_labels = gts_labels[:trainSize]
    test_masks = all_masks[trainSize:]
    image = np.transpose(image, (0,3,1,2))
    
    # split label
    train_labels = {'boxes':gts_boxes, 'labels':gts_labels}
    labels = {'boxes':gts_boxes, 'labels':gts_labels}
    test_labels = {'boxes':test_gts_boxes, 'labels':test_gts_labels}
    
    eps = 0.5
    print(eps)

    DIR = DIR_BASE
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    attack = DPatch(
        mdl,
        patch_shape=(16, 16, 3),
        learning_rate=eps,
        max_iter=12,
        batch_size=1,
        verbose=False,
    )

    patch = np.load("patches/aware/frcnn/"+cate+"/patch_{}.npy".format(eps))
    mdl._model.training = False

    train_pert_images = image 
    test_pert_images = test_images
   
    test_pert_images = np.transpose(test_pert_images, (0,3,1,2))
    train_pert_images = np.transpose(train_pert_images, (0,3,1,2))

    mdl._model.training = False
    origin_iou, patched_iou = 0, 0
    
    cnt = 0
    for j in range(test_images.shape[0]):
        y_target = dict()
        y_target['boxes'] = torch.from_numpy(test_labels['boxes'][j]).type(torch.float).cuda()
        y_target["labels"] = torch.from_numpy(test_labels['labels'][j]).cuda()
        y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(test_labels['boxes'][j])])).type(torch.int64).cuda()
        # (900, 3, 224, 224)
        pert_images, new_boxes = attack.apply_no_patch(torch.Tensor(test_images[j:j+1]).cuda(), 
                                                        patch_external=torch.Tensor(patch).cuda(), 
                                                        gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda(), 
                                                        corrupt_type=5, 
                                                        masks=test_masks[j],
                                                        clear=test_clear)
        if model_name == "frcnn":
            pert_predictions = mdl.predict(x=pert_images)
        else:
            pert_predictions = mdl.predict(x=np.transpose(pert_images, (0,3,1,2))/255.)
        # for json, 0 threshold
        try:
            # for plot, 0.5 threshold        
            predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", cls=CLS)
            if count > 0:
                plot_image_with_boxes(img=np.ascontiguousarray(pert_images[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, cls=CLS, gt_boxes=new_boxes) #test_labels['boxes'][j])
                plt.savefig(DIR+"/frost_pert_test_image_{}.png".format(j))
                iscrowd = torch.zeros(len(new_boxes))
                predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]

                ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                origin_iou += (ious > 0.5).sum()
        except Exception as e:
            pass

        cnt += len(test_labels['boxes'][j])
        patched_images, new_boxes, transferred_masks = attack.apply_multi_affine(torch.Tensor(test_images[j:j+1]).cuda(), 
                                                                                    patch_external=torch.Tensor(patch).cuda(), 
                                                                                    gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda(), 
                                                                                    masks=test_masks[j],
                                                                                    clear=test_clear)
        if model_name == "frcnn":
            patch_predictions = mdl.predict(patched_images)
        else:
            patch_predictions = mdl.predict(np.transpose(patched_images.astype(np.float32), (0,3,1,2))/255.)
        # for ii in range(len(transferred_masks)):
        #     plt.imshow(transferred_masks[ii].T)
        #     plt.savefig("affine_frcnn/masks{}_{}.png".format(j, ii))
        try:
            predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions[0], name="Patched", cls=CLS)
            if count > 0:
                plot_image_with_boxes(img=np.ascontiguousarray(patched_images[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, cls=CLS, gt_boxes=new_boxes) #test_labels['boxes'][j])
                plt.savefig(DIR+"/frost_patched_test_image_{}.png".format(j))
                iscrowd = torch.zeros(len(new_boxes))
                predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]

                ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                patched_iou += (ious > 0.5).sum()                
        except Exception as e:
            pass
    
    print(cate, ",", prefix, "affine test, image saves in:", DIR)
    print("IoU 0.5 acc: Origin", origin_iou/cnt, ", Patched", patched_iou/cnt, origin_iou, patched_iou)
    
if __name__ == "__main__":
    main()
