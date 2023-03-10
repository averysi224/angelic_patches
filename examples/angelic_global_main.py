import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchFasterRCNN
from art.estimators.object_detection import PyTorchSSD
from art.attacks.evasion import DPatch
from pycocotools import mask as maskUtils

import pdb
import os
import json
import copy

import dsld
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import argparse
import os
from pkg_resources import resource_filename

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')

# Batch size
train_batch_size = 1
im_length = 224
model_name = "frcnn"

cate_list = {"person":1, "bus":6, "bottle": 44, "cup": 47, "bowl":51, "chair":62, "laptop":73}

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
    
def load_normalized_bases():
    images = os.listdir('frost')
    frost_bases = []
    for im in images:
        image = cv2.imread('frost/' + im)[..., [2, 1, 0]]
        image = np.expand_dims(image, 0)
        frost_bases.append(image)
    return frost_bases

def get_loss(mdl, x, y):
    mdl._model.train()
    image_tensor_list = list()
    for i in range(x.shape[0]):
        if mdl.clip_values is not None:
            img = x[i] / mdl.clip_values[1]
        else:
            img = x[i]
        image_tensor_list.append(img)
    loss = mdl._model(image_tensor_list, [y])
    for loss_type in mdl.attack_losses:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss

def append_loss_history(loss_history, output, num):
    for loss in loss_history.keys():
        loss_history[loss] += output[loss] / num
    return loss_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cate', default="bus", help = "target test category")
    parser.add_argument('--clear', action="store_true", help = "test without corruption, default with corruption.")
    parser.add_argument('--agnostic', action="store_true", help = "corruption-agnostic patch, default aware.")
    parser.add_argument('--train_patch', action="store_true", help = "If yes, train; default, use provided patches.")
    parser.add_argument('--partial', action="store_true", help = "If yes, apply patch on some of the objects.")
    parser.add_argument('--severity', default=1, help = "corruption level.")
    parser.add_argument('--randplace', action="store_true", help = "If yes, apply patch not in center.")
    parser.add_argument('--visualize', action="store_true", help = "If yes, save detection results.")
    parser.add_argument('--coco_path', default='/data5/wenwens/coco2017/train2017', help = "path of COCO dataset")
    args = parser.parse_args()
    print(args)

    cate = args.cate # category name
    test_clear = args.clear
    CLS=cate_list[cate]  # category label

    suffix = "agnostic" if args.agnostic else "aware"
    suffix += "_clear" if args.clear else "_corrupt"
    suffix += "_partial" if args.clear else ""
    suffix += "_randplace" if args.clear else ""

    DIR_BASE='results/{}/{}_{}'.format(model_name, cate, suffix) 
    aware = not args.agnostic
    train_patch = args.train_patch
    severity = args.severity
    load_size = 2000 if train_patch else 400
    ratio = 0.9 if train_patch else 0.1
    partial_test = args.partial
    rand_place = args.randplace
    path2data = args.coco_path # path to coco dataset
    visualize = args.visualize
    path2json = '../coco-manager/instances_'+cate+'_train2017.json' # filtered single category json

    if aware: 
        severity = 3

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bases = load_normalized_bases()

    image = []
    gts_boxes = []
    gts_labels = []
    img_ids = []

    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations, _] in enumerate(data_loader):
        if len(image) >=load_size:    # if too much data, use first 4000
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

            except Exception as e:
                pass
    
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
    image = np.transpose(image, (0,3,1,2))
    
    # split label
    train_labels = {'boxes':gts_boxes, 'labels':gts_labels}
    labels = {'boxes':gts_boxes, 'labels':gts_labels}
    test_labels = {'boxes':test_gts_boxes, 'labels':test_gts_labels}
    
    eps = 0.5
    DIR = DIR_BASE + str(eps)

    print("Results will be save in", DIR)
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

    if train_patch:
        print("eps =", eps)
        for j in range(image.shape[0]):
            y_target = dict()
            y_target['boxes'] = torch.from_numpy(train_labels['boxes'][j]).type(torch.float).cuda()
            y_target["labels"] = torch.from_numpy(train_labels['labels'][j]).cuda()
            y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(train_labels['boxes'][j])])).type(torch.int64).cuda()
            ####################
            if aware:
                train_pert_image = getattr(attack, "frost")(np.transpose(copy.deepcopy(image[j:j+1]), (0,2,3,1)), bases, severity=severity).astype(np.float32)
                train_pert_image = np.transpose(train_pert_image, [0,3,1,2])
            else:
                train_pert_image = image[j:j+1]
            origin_loss = get_loss(mdl, copy.deepcopy(torch.Tensor(train_pert_image).cuda()), copy.deepcopy(y_target))
            original_loss_history = append_loss_history(original_loss_history, origin_loss, image.shape[0])
        prefix = "train pert image loss: " if aware else "train clear image loss: "
        print(prefix, original_loss_history)  # train pert image  
        patch, acc_loss1, acc_loss2 = attack.generate(x=image, target_label=CLS, labels=labels, aware=aware, model_name=model_name)
        plt.axis("off")
        plt.title("Adversarial Patch")
        plt.imshow(np.transpose(patch,(0,2,3,1))[0].astype(np.uint8), interpolation="nearest")
        plt.savefig(DIR+"/patch_{}.png".format(eps))
        np.save(os.path.join(DIR+"/patch_{}".format(eps)), patch)
        plt.close()
    else:
        if aware:
            # patch = np.load("patches/aware/{}/{}/patch_0.5.npy".format(model_name, cate))
            patch = np.load("patches/cross/{}/patch_0.5.npy".format(cate))
        else:
            patch = np.load("patches/agnostic/{}_{}/patch_0.5.npy".format(model_name, cate))

    mdl._model.training = False

    pert_data, patch_data = [], []    # prepare data for IoU json 
    original_loss_history = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}
    
    mdl._model.training = False
    for c1 in range(len(attack.cdict)):
        crr=attack.cdict[c1]
        if aware and c1 > 0: 
            break
        if test_clear:
            print(cate, crr, "patch, clear test.")
        else:
            print(cate, crr, "attack, severity =", severity)
        cnt = 0
        origin_iou, patched_iou, patched_cnt = 0, 0, 0
        for j in range(test_images.shape[0]):
            y_target = dict()
            y_target['boxes'] = torch.from_numpy(test_labels['boxes'][j]).type(torch.float).cuda()
            y_target["labels"] = torch.from_numpy(test_labels['labels'][j]).cuda()
            y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(test_labels['boxes'][j])])).type(torch.int64).cuda()
            if test_clear:
                x_tmp = np.transpose(copy.deepcopy(test_images[j:j+1]), (0,2,3,1))
            else:
                if c1 > 0:
                    # actual input (1, 224, 224, 3)
                    x_tmp = getattr(attack, crr)(np.transpose(copy.deepcopy(test_images[j:j+1]), (0,2,3,1)), severity=severity).astype(np.float32)
                else:
                    # (1, 3, 224, 224)
                    # actual input (224, 224, 3, 1)
                    x_tmp = getattr(attack, crr)(np.transpose(copy.deepcopy(test_images[j:j+1]), (0,2,3,1)), bases, severity=severity).astype(np.float32)
                    # output (1, 224, 224, 3)
            if model_name == "frcnn":
                pert_predictions = mdl.predict(x=x_tmp)
            else:
                pert_predictions = mdl.predict(x=np.transpose(x_tmp, (0,3,1,2))/255.)
            
            # for json, 0 threshold
            try:
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", cls=CLS, thresh=0.0)
                if count > 0:
                    # json results
                    for kk in range(len(predictions_boxes)):
                        res = dict()
                        res['image_id'] = test_ids[j].item()
                        res['category_id'] = 1 #CLS
                        b = predictions_boxes[kk]
                        res['bbox'] = np.array([b[0][0], b[0][1], b[1][0]-b[0][0], b[1][1]-b[0][1]])
                        res['score'] = predictions_scores[kk]
                        pert_data.append(res)
                # for plot, 0.5 threshold        
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", cls=CLS)
                if count > 0:
                    if visualize:
                        plot_image_with_boxes(img=np.ascontiguousarray(x_tmp[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j], cls=CLS)
                        plt.savefig(DIR+"/" + crr + "_pert_test_image_{}.png".format(j))
                    new_boxes = copy.deepcopy(test_labels['boxes'][j])
                    iscrowd = torch.zeros(len(new_boxes))
                    predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                    predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                    new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]
                    ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                    origin_iou += (ious > 0.5).sum()
            except Exception as e:
                pass

            cnt += len(test_labels['boxes'][j])
            # actual input (40, 3, 224, 224)
            pdb.set_trace()
            patched_images, rand_n = attack.apply_multi_patch(torch.Tensor(test_images[j:j+1]).cuda(), 
                                                    patch_external=torch.Tensor(patch).cuda(), 
                                                    gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda(), 
                                                    corrupt_type=c1,
                                                    severity=severity,
                                                    clear=test_clear,
                                                    partial=partial_test,
                                                    rp=rand_place)
            patched_images = patched_images.astype(np.float32)
            patched_cnt += rand_n
            if model_name == "frcnn":
                patch_predictions = mdl.predict(patched_images)
            else:
                patch_predictions = mdl.predict(np.transpose(patched_images.astype(np.float32), (0,3,1,2))/255.)
            
            try:
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions[0], name="Patched", cls=CLS, thresh=0.0)
                if count > 0:
                    # json results
                    for kk in range(len(predictions_boxes)):
                        res = dict()
                        res['image_id'] = test_ids[j].item()
                        res['category_id'] = 1 #CLS
                        b = predictions_boxes[kk]
                        res['bbox'] = np.array([b[0][0], b[0][1], b[1][0]-b[0][0], b[1][1]-b[0][1]])
                        res['score'] = predictions_scores[kk]
                        patch_data.append(res)
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions[0], name="Patched", cls=CLS)
                if count > 0:
                    if visualize:
                        plot_image_with_boxes(img=np.ascontiguousarray(patched_images[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j], cls=CLS)
                        plt.savefig(DIR+"/" + crr + "_patched_test_image_{}.png".format(j))
                    new_boxes = copy.deepcopy(test_labels['boxes'][j])
                    iscrowd = torch.zeros(len(new_boxes))
                    predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                    predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                    new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]

                    ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                    patched_iou += (ious > 0.5).sum()
            except Exception as e:
                pass
        
        # print(origin_iou, patched_iou)
        if partial_test:
            print("0.5 IoU high Conf Acc: partial ", origin_iou/cnt, ", Patched", patched_iou/cnt, ", Correct / Patched", patched_iou/patched_cnt)
        else:
            print("IoU 0.5 Acc: Origin", origin_iou/cnt, ", Patched", patched_iou/cnt, origin_iou, patched_iou)
        
        with open(DIR+"/" + crr + "_pert_res.json", "w") as out_file:
                out_file.write(json.dumps(pert_data, cls=NumpyEncoder))
        with open(DIR+"/" + crr + "_patch_res.json", "w") as out_file:
            out_file.write(json.dumps(patch_data, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
