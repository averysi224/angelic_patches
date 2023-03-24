import torch
import torchvision
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

from art.estimators.object_detection import PyTorchFasterRCNN
from pytorch_ssd import PyTorchSSD
from apatch import AngelicPatch
from tqdm import tqdm
import os
import json
import copy
import dsld
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.multiprocessing.set_sharing_strategy('file_system')

# Batch size
train_batch_size = 1
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
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            gbox = [(gt_boxes[i][0], gt_boxes[i][1]), (gt_boxes[i][2], gt_boxes[i][3])]
            # Draw Rectangle with the coordinates
            cv2.rectangle(img, gbox[0], gbox[1], color=(0, 255, 255), thickness=1)

    plt.axis("off")
    img = img[:, :, ::-1]
    
    plt.imshow(img.astype(np.uint8), interpolation="nearest")

length = 224

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
    for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss

def append_loss_history(loss_history, output, num):
    for loss in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss_history[loss] += output[loss] / num
    return loss_history

def main():
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    ssd = PyTorchSSD(
        clip_values=(0, 255), model=model, attack_losses=["bbox_regression", "classification"]
    )  
    parser = argparse.ArgumentParser()
    parser.add_argument('--cate', default="bus", help = "target test category")
    parser.add_argument('--model_name', default="retina", help = "choose from fcos and retinanet.")
    parser.add_argument('--coco_path', default='/data5/wenwens/coco2017/train2017', help = "path of COCO dataset")
    parser.add_argument('--train_patch', action="store_true", help = "If yes, train; default, use provided patches.")
    parser.add_argument('--visualize', action="store_true", help = "If yes, save detection results.")
    args = parser.parse_args()

    cate = args.cate # category name
    train_patch = args.train_patch
    load_size = 2000 if train_patch else 200
    ratio = 0.9 if train_patch else 0.1
    CLS=cate_list[cate]  # category label
    cocoCLS=CLS
    model_name = args.model_name

    DIR_BASE = 'results/{}/cross_{}'.format(model_name, cate) 
    path2data =  args.coco_path # path to coco dataset
    path2json = 'category_json/instances_'+cate+'_train2017.json'  # filtered single category json

    # create own Dataset
    my_dataset = dsld.myOwnDataset(root=path2data,
                            annotation=path2json,
                            im_length=length,
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

    eps=0.5
    attack = AngelicPatch(
        ssd,
        frcnn,
        patch_shape=(16, 16, 3),
        learning_rate=eps,
        max_iter=12,
        batch_size=1,
        verbose=False,
        im_length=224,
    )

    image = []
    gts_boxes = []
    gts_labels = []
    pert_image = []
    img_ids = []

    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations, _] in enumerate(data_loader):
        if len(image) >=load_size:    
            break
        if idx % 1000 == 0:
            print(idx)
        
        imgs = torch.stack(imgs)
        imgs = imgs.permute(0,2,3,1)
        imgs = imgs.numpy()[:, :,:, ::-1]*255.0
        if np.max(imgs) == 0:
            continue
        
        pert_imgs = getattr(attack, "frost")(imgs, bases, severity=3)

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
                    gts_labels.append(gt_class*cocoCLS)
                    img_ids.append(annotations[i]["image_id"])
                    image.append(imgs[i])

            except Exception as e:
                pass

    print("all", idx)
    image = np.stack(image)
    pert_image = np.stack(pert_image)
    print(image.shape)
    print("generate")
    labels = {'boxes':gts_boxes, 'labels':gts_labels}
    # split dataset
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
    
    DIR = DIR_BASE + str(eps)
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    print(eps)
    if args.train_patch:
        patch, acc_loss1, acc_loss2 = attack.generate_cross(x=image, target_label=CLS, labels=labels)
    else:
        patch = np.load("patches/cross/"+cate+"/patch_{}.npy".format(eps))
    
    plt.axis("off")
    plt.title("Adversarial Patch")
    plt.imshow(np.transpose(patch,(0,2,3,1))[0].astype(np.uint8), interpolation="nearest")
    plt.savefig(DIR+"/patch_{}.png".format(eps))
    np.save(os.path.join(DIR+"/patch_{}".format(eps)), patch)

    if model_name == "retina":
        yolo5 = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        yolo5.eval()
    elif model_name == "fcos":
        yolo5 = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
        yolo5.eval()
    else:
        raise NotImplementedError("For Frcnn & SSD cross-model transferability test, please use angelic_global_main.py")

    frcnn._model.training = False
    origin_iou, patched_iou = 0, 0
    cnt = 0

    print("Warning, this test may be slow. Please be patient.")
    for j in tqdm(range(test_images.shape[0])):
        y_target = dict()
        y_target['boxes'] = torch.from_numpy(test_labels['boxes'][j]).type(torch.float).cuda()
        y_target["labels"] = torch.from_numpy(test_labels['labels'][j]).cuda()
        y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(test_labels['boxes'][j])])).type(torch.int64).cuda()
        with torch.no_grad():
            results = yolo5(copy.deepcopy(torch.Tensor(test_pert_images[j:j+1]))/255.)[0]
        pert_predictions = {}
        pert_predictions['boxes'] = results['boxes'].detach().cpu().numpy()
        pert_predictions['labels'] = results['labels'].detach().cpu().numpy()
        pert_predictions['scores'] = results['scores'].detach().cpu().numpy()
        
        try:
            predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions, name="Perturbed", cls=CLS)
            if count > 0:
                if args.visualize:
                    plot_image_with_boxes(img=np.ascontiguousarray(np.transpose(test_pert_images[j],(1,2,0)), dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j], cls=CLS)
                    plt.savefig(DIR+"/pert_test_image_{}.png".format(j))
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
        patched_images, _ = attack.apply_multi_patch(torch.Tensor(test_images[j:j+1]).cuda(), patch_external=torch.Tensor(patch).cuda(), gts_boxes=torch.Tensor(test_gts_boxes[j]).cuda(),corrupt_type=0,severity=3)
        with torch.no_grad():
            results = yolo5(copy.deepcopy(torch.Tensor(patched_images).permute(0,3,1,2))/255.)[0]
            
        patch_predictions = {}
        patch_predictions['boxes'] = results['boxes'].detach().cpu().numpy()
        patch_predictions['labels'] = results['labels'].detach().cpu().numpy()
        patch_predictions['scores'] = results['scores'].detach().cpu().numpy()
        try:
            predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions, name="Patched", cls=CLS)                
            if count > 0:
                if args.visualize:
                    plot_image_with_boxes(img=np.ascontiguousarray(patched_images[0]), boxes=predictions_boxes, pred_cls=predictions_class, gt_boxes=test_labels['boxes'][j], cls=CLS)
                    plt.savefig(DIR+"/patched_test_image_{}.png".format(j))
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
    print(origin_iou/cnt, ",", patched_iou/cnt)

if __name__ == "__main__":
    main()
