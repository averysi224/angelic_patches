import torch
import torchvision
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchSSD
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DPatch
from pycocotools import mask as maskUtils

import torch
import dsssd
import pdb
import os
import json
import copy

import torchvision

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.multiprocessing.set_sharing_strategy('file_system')

# Batch size
train_batch_size = 1
CLS=1  # category label
cate = 'person'
DIR_BASE='testtest'  # savedir base

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

def extract_predictions(predictions_, name, cls=CLS, thresh=0.5, eps=1):
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

def plot_image_with_boxes(img, boxes, pred_cls, cls=CLS, gt_boxes=None):
    text_size = 1
    text_th = 2
    rect_th = 2

    for i in range(len(boxes)):
        if pred_cls[i] == cls:
            # Draw Rectangle with the coordinates, green
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=1)
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            gbox = [(gt_boxes[i][0], gt_boxes[i][1]), (gt_boxes[i][2], gt_boxes[i][3])]
            # Draw Rectangle with the coordinates
            cv2.rectangle(img, gbox[0], gbox[1], color=(0, 255, 255), thickness=1)

    plt.axis("off")
    img = img[:, :, ::-1]
    
    plt.imshow(img.astype(np.uint8), interpolation="nearest")

length = 300

def frost(x, bases, coords=None, severity=3):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(len(bases))
    frost = bases[idx]
    x_start, y_start = np.random.randint(0, frost.shape[1] - length), np.random.randint(0, frost.shape[2] - length)
    frost_piece = frost[:, x_start:x_start + length, y_start:y_start + length, :]
    if coords is None:
        return np.clip(c[0] * x + c[1] * frost_piece, 0, 255)
    else:
        patch_frosts = []
        for i in range(len(coords)):
            patch_frosts.append(frost_piece[:, coords[i][0]:coords[i][1], coords[i][2]:coords[i][3], :])
        return np.clip(c[0] * x + c[1] * frost_piece, 0, 255), patch_frosts

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x1 = np.reshape(x, [1,-1,3])
    means = np.mean(x1, axis=1)
    return np.clip((x - means) * c + means, 0, 255), means

def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def brightness(x, coords=None, severity=1):  #(w, h, c) # 3 224 224
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.clip((1 + c) * x, 0, 255.)
    
    return x, 1+c

def fog(x, coords=None, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    change = c[0] * plasma_fractal(wibbledecay=c[1])[:length, :length][..., np.newaxis]
    change = np.repeat(change, [3], axis=2) * 255
    max_val = x.max() / 255.
    xx = x + change
    xx = np.clip(xx * max_val / (max_val + c[0]), 0, 255) 
    if coords is None:
        return xx
    else:
        patch_fogs = []
        for i in range(len(coords)):
            patch_fogs.append(change[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]])

        return xx, patch_fogs, max_val

def load_normalized_bases():
    images = os.listdir('frost');
    frost_bases = []
    for im in images:
        image = cv2.imread('frost/' + im)[..., [2, 1, 0]]
        image = np.expand_dims(image, 0)
        frost_bases.append(image)
    return frost_bases

def get_loss(frcnn, x, y):
    frcnn._model.train()
    image_tensor_list = list()
    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = x[i] / frcnn.clip_values[1]
        else:
            img = x[i]
        image_tensor_list.append(img)
    loss = frcnn._model(image_tensor_list, [y])
    for loss_type in ["bbox_regression", "classification"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss

def append_loss_history(loss_history, output, num):
    for loss in ["bbox_regression", "classification"]:
        loss_history[loss] += output[loss] / num
    return loss_history

def init_loss_history(num=1):
    loss = dict()
    for loss_type in ["bbox_regression", "classification"]:
        loss[loss_type] = num
    return loss

def min_loss_history(loss_history_1, loss_history_2=None):
    if loss_history_2 != None:
        for loss in ["bbox_regression", "classification"]:
            loss_history_1[loss] = min(loss_history_1[loss], loss_history_2[loss])
    return loss_history_1

def main():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    ssd = PyTorchSSD(
        clip_values=(0, 255), model=model, attack_losses=["bbox_regression", "classification"]
    )

    path2data = '/data5/wenwens/coco2017/train2017'
    path2json = '../coco-manager/instances_'+cate+'_train2017.json'  # 

    # create own Dataset
    my_dataset = dsssd.SSDDataset(root=path2data,
                            annotation=path2json,
                            transforms=dsssd.get_transform()
                            )

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
    pert_image = []
    img_ids = []

    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations, masks] in enumerate(data_loader):
        if len(image) >= 500:    # if too much data, use first 4000
            break
        if idx % 1000 == 0:
            print(idx)
        
        imgs = torch.stack(imgs)
        imgs = imgs.permute(0,2,3,1)
        imgs = imgs.numpy()[:, :,:, ::-1]*255.0
        if np.max(imgs) == 0:
            continue

        pert_imgs = frost(imgs, bases)
        # pert_imgs, _ = contrast(pert_imgs)
        # pert_imgs = fog(imgs)
        # pert_imgs, _ = brightness(imgs)
        pert_imgs = pert_imgs.astype(np.float32)

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
                    pert_image.append(pert_imgs[i])
                    gts_boxes.append(gt_boxes)
                    gts_labels.append(gt_class*CLS)
                    img_ids.append(annotations[i]["image_id"])
                    image.append(imgs[i])

            except Exception as e:
                pass

    image = np.stack(image)
    pert_image = np.stack(pert_image)
    print(image.shape)
    print("generate")
    labels = {'boxes':gts_boxes, 'labels':gts_labels}
    # split dataset
    ratio = 0.5
    trainSize = int(image.shape[0] * ratio)
    testSize = (image.shape[0] - trainSize)
    test_images = image[trainSize:,]
    test_gts_boxes = gts_boxes[trainSize:]
    test_gts_labels = gts_labels[trainSize:]
    test_ids = img_ids[trainSize:]
    train_pert_images = pert_image[:trainSize,]
    test_pert_images = pert_image[trainSize:,]
    test_labels = {'boxes':test_gts_boxes, 'labels':test_gts_labels}
    image = image[:trainSize,]
    gts_boxes = gts_boxes[:trainSize]
    gts_labels = gts_labels[:trainSize]
    train_labels = {'boxes':gts_boxes, 'labels':gts_labels}
    # (N,W,H,C) -> (N,C,W,H)
    image = np.transpose(image, (0,3,1,2))
    test_images = np.transpose(test_images, (0,3,1,2))
    test_pert_images = np.transpose(test_pert_images, (0,3,1,2))
    train_pert_images = np.transpose(train_pert_images, (0,3,1,2))
    
    for eps in [0.5]:
        DIR = DIR_BASE + str(eps)
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        attack = DPatch(
            ssd,
            ssd,
            patch_shape=(16, 16, 3),
            learning_rate=eps,
            max_iter=12,
            batch_size=1,
            verbose=False,
        )
        print(eps)
        # patch, acc_loss1, acc_loss2 = attack.generate(x=image, target_label=CLS, labels=labels)
        # patch = np.load("patches/aware/frcnn/"+cate+"/patch_{}.npy".format(eps))
        patch = np.load("global_results_full/"+cate+"_cross_focs0.5/patch_{}.npy".format(eps))
        pert_data, patch_data = [], []    # prepare data for IoU json

        loss_history = {"bbox_regression": 0, "classification": 0}
        original_loss_history = {"bbox_regression": 0, "classification": 0}
        origin_iou, patched_iou = 0, 0
        cnt = 0
        for j in range(test_images.shape[0]):
            y_target = dict()
            y_target['boxes'] = torch.from_numpy(test_labels['boxes'][j]).type(torch.float).cuda()
            y_target["labels"] = torch.from_numpy(test_labels['labels'][j]).cuda()
            y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(test_labels['boxes'][j])])).type(torch.int64).cuda()
            
            # origin_loss = get_loss(ssd, copy.deepcopy(torch.Tensor(test_pert_images[j:j+1]).cuda()), copy.deepcopy(y_target))
            # original_loss_history = append_loss_history(original_loss_history, origin_loss, test_pert_images.shape[0])
            pert_predictions = ssd.predict(x=test_pert_images[j:j+1]/255.)
            try:
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", thresh=0.0, cls=CLS)
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
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", cls=CLS)
                if count > 0:
                    # plot_image_with_boxes(img=np.ascontiguousarray(np.transpose(test_pert_images[j],(1,2,0)), dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j])
                    # plt.savefig(DIR+"/pert_test_image_{}.png".format(j))
                    new_boxes = copy.deepcopy(test_labels['boxes'][j])
                    iscrowd = torch.zeros(len(new_boxes))
                    predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                    predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                    new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]

                    ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                    origin_iou += (ious > 0.5).sum()
                # print('\n')
            except Exception as e:
                pass

            cnt += len(test_labels['boxes'][j])
            # patched_images = attack.apply_no_corruption(torch.Tensor(test_images[j:j+1]).cuda(), patch_external=torch.Tensor(patch).cuda(), gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda())
            patched_images = attack.apply_multi_patch(torch.Tensor(test_images[j:j+1]).cuda(), patch_external=torch.Tensor(patch).cuda(), gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda())
            loss = get_loss(ssd, copy.deepcopy(patched_images), copy.deepcopy(y_target))
            loss_history = append_loss_history(loss_history, loss, test_images.shape[0])
            patched_images = patched_images.cpu().numpy()
            patch_predictions = ssd.predict(patched_images/255., y_target)
            try:
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions[0], name="Patched", thresh=0.0, cls=CLS)
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
                    # plot_image_with_boxes(img=np.ascontiguousarray(np.transpose(patched_images[0],(1,2,0)), dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j])
                    # plt.savefig(DIR+"/patched_test_image_{}.png".format(j))
                    new_boxes = copy.deepcopy(test_labels['boxes'][j])
                    iscrowd = torch.zeros(len(new_boxes))
                    predictions_boxes = np.reshape(np.array(predictions_boxes), (-1, 4))
                    predictions_boxes[:, 2:] = predictions_boxes[:, 2:] - predictions_boxes[:, :2]
                    new_boxes[:, 2:] = new_boxes[:, 2:]  - new_boxes[:, :2]

                    ious = maskUtils.iou(predictions_boxes, new_boxes, iscrowd)
                    patched_iou += (ious > 0.5).sum()
            except Exception as e:
                pass
        
        print(cate)
        print("patched loss:", loss_history, cnt)
        print("original loss:", original_loss_history)
        print(origin_iou/cnt, ",", patched_iou/cnt)
        
        with open(DIR+"/pert_res.json", "w") as out_file:
            out_file.write(json.dumps(pert_data, cls=NumpyEncoder))
        with open(DIR+"/patch_res.json", "w") as out_file:
            out_file.write(json.dumps(patch_data, cls=NumpyEncoder))

        with open(DIR+"/loss_history_{}.json".format(eps), "a") as file:
            file.write(json.dumps(loss_history))
            file.write(json.dumps(original_loss_history))
            file.write(json.dumps(cnt))
        np.save(os.path.join(DIR+"/patch_{}".format(eps)), patch)

if __name__ == "__main__":
    main()
