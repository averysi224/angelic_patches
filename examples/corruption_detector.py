import cv2
import numpy as np
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DPatch
import torch
import dsld
import pdb
import os
import json
import copy

import torchvision

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.multiprocessing.set_sharing_strategy('file_system')

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

# Batch size
train_batch_size = 1
CLS=6
DIR="bus1.0/"
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def extract_predictions(predictions_, name, cls=CLS, eps=1):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]

    # Get the predicted bounding boxes
    predictions_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(predictions_["boxes"].astype(int))]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])

    # Get a list of index with score greater than threshold
    threshold = 0.5

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

    if count > 0:
        pred_clses = list( predictions_class[i] for i in tgt_idx)
        pred_scores = list( predictions_score[i] for i in tgt_idx)
        perform = {"name": name, "pred_class": pred_clses, "pred_score" : np.array(pred_scores)}
        # pdb.set_trace()
        with open(DIR+"loss_history_{}.json".format(eps), "a") as file:
            file.write(json.dumps(perform, cls=NumpyEncoder))
        # print(name + " results")
        # print("predicted classes:", list( predictions_class[i] for i in tgt_idx))
        # print("predicted score:", list( predictions_score[i] for i in tgt_idx))

    # return predictions_class, predictions_boxes, predictions_score
    return predictions_["labels"], predictions_boxes, predictions_score, count

def plot_image_with_boxes(img, boxes, pred_cls, cls=CLS):
    text_size = 1
    text_th = 2
    rect_th = 2
    #pdb.set_trace()
    for i in range(len(boxes)):
        if pred_cls[i] == cls:
            # pdb.set_trace()
            # Draw Rectangle with the coordinates
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(255, 255, 255), thickness=rect_th)

            # Write the prediction class
            cv2.putText(img, str(pred_cls[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (125, 125, 125), thickness=text_th)

    plt.axis("off")
    img = img[:, :, ::-1]
    
    plt.imshow(img.astype(np.uint8), interpolation="nearest")

def load_normalized_bases():
    images = os.listdir('frost');
    frost_bases = []
    for im in images:
        image = cv2.imread('frost/' + im)[..., [2, 1, 0]]
        image = np.expand_dims(image, 0)
        frost_bases.append(image)
    return frost_bases

def frost(x, bases, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    frost = bases[idx]
    patch_frosts = []
    x_start, y_start = np.random.randint(0, frost.shape[1] - 224), np.random.randint(0, frost.shape[2] - 224)
    frost_piece = frost[:, x_start:x_start + 224, y_start:y_start + 224, :]
    
    return np.clip(c[0] * x + c[1] * frost_piece, 0, 255)

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x1 = np.reshape(x, [1,-1,3])
    means = np.mean(x1, axis=1)
    return np.clip((x - means) * c + means, 0, 255)

def get_loss(frcnn, x, y):
    frcnn._model.train()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor_list = list()

    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            img = transform(x[i]).to(frcnn._device)
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

    path2data = '/data5/wenwens/coco2017/train2017'
    path2json = 'examples/instances_bus_train2017_clean.json'

    # create own Dataset
    my_dataset = dsld.myOwnDataset(root=path2data,
                            annotation=path2json,
                            transforms=dsld.get_transform()
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
        
    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations] in enumerate(data_loader):
        #if len(image) >= 100:
        #   break
        # if idx % 1000 == 0:
        #     print(idx)
        imgs = torch.stack(imgs)
        imgs = imgs.permute(0,2,3,1)
        imgs = imgs.numpy()[:, :,:, ::-1]*255.0

        predictions = frcnn.predict(x=imgs)

        pert_imgs = frost(imgs, bases)
        pert_imgs = contrast(pert_imgs)
        pert_imgs = pert_imgs.astype(np.float32)

        # pert_predictions = frcnn.predict(x=pert_imgs)

        for i in range(imgs.shape[0]):
            # Process predictions
            try:
                # Plot predictions
                gt_class, gt_boxes = annotations[i]['labels'].numpy(), annotations[i]['boxes'].numpy().astype(int)
                count = np.sum(gt_class == 1)
                #pdb.set_trace()
                #if count > 1:
                #    predictions_boxes = []
                #    for ii in range(len(gt_class)):
                #        predictions_boxes.append([(int(gt_boxes[ii,0]), int(gt_boxes[ii,1])), (int(gt_boxes[ii,2]), int(gt_boxes[ii,3]))])
                #    ordered_classes = np.array(range(len(gt_class)))
                #    plot_image_with_boxes(img=imgs[i].copy(), boxes=predictions_boxes, pred_cls=ordered_classes)
                #    plt.savefig("sort/image_{}.png".format(annotations[i]["image_id"].item()))

                if count > 0:
                    pert_image.append(pert_imgs[i])
                    gts_boxes.append(gt_boxes)
                    gts_labels.append(gt_class*CLS)
                    # if len(image) % 10 == 0:
                    #     predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(predictions[i], name="Original", cls=6)
                    #     plot_image_with_boxes(img=imgs[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class)
                    #     plt.savefig("good/original_test_image_{}.png".format(len(image)))
                    image.append(imgs[i])

            except Exception as e:
                pass

        # visualize_predictions(pert_imgs, pert_predictions, annotations, idx, True)
        # for i in range(pert_imgs.shape[0]):
            # Process predictions
            # try:
            #     # predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[i], name="Perturbed", cls=6)
            #     # if count > 0:
            #     #     print('count', count, len(image))
            # except Exception as e:
            #     pass

    image = np.stack(image)
    pert_image = np.stack(pert_image)
    print(image.shape)
    s=0
    for label in gts_labels:
        if len(label) < 3:
         s += 1
    #pdb.set_trace()
    print("toaster", s)
    labels = {'boxes':gts_boxes, 'labels':gts_labels}

    # split dataset
    ratio = 0.8
    trainSize = int(image.shape[0] * ratio)
    testSize = (image.shape[0] - trainSize)
    test_images = image[trainSize:,]
    test_gts_boxes = gts_boxes[trainSize:]
    test_gts_labels = gts_labels[trainSize:]
    test_pert_images = pert_image[trainSize:,]
    test_labels = {'boxes':test_gts_boxes, 'labels':test_gts_labels}
    image = image[:trainSize,]
    gts_boxes = gts_boxes[:trainSize]
    gts_labels = gts_labels[:trainSize]
    pert_image = pert_image[:trainSize,]

    for eps in [1.0]:
        attack = DPatch(
            frcnn,
            patch_shape=(16, 16, 3),
            learning_rate=eps,
            max_iter=1,
            batch_size=1,
            verbose=False,
        )
        print(eps)
        patch = attack.generate(x=image, target_label=CLS, labels=labels)

        #TODO compute ground truth count

        plt.axis("off")
        plt.title("Adversarial Patch")
        plt.imshow(patch.astype(np.uint8), interpolation="nearest")
        plt.savefig("patch_{}.png".format(eps))

        loss_history = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}
        original_loss_history = {"loss_classifier": 0, "loss_box_reg": 0, "loss_objectness": 0, "loss_rpn_box_reg": 0}

        cnt = 0
        for j in range(test_images.shape[0]):
            y_target = dict()
            y_target['boxes'] = torch.from_numpy(test_labels['boxes'][j]).type(torch.float).cuda()
            y_target["labels"] = torch.from_numpy(test_labels['labels'][j]).cuda()
            y_target["scores"] = torch.from_numpy(1.0 * np.ones([len(test_labels['boxes'][j])])).type(torch.int64).cuda()

            origin_loss = get_loss(frcnn, copy.deepcopy(test_pert_images[j:j+1]), y_target)
            original_loss_history = append_loss_history(original_loss_history, origin_loss, test_pert_images.shape[0])

            pert_predictions = frcnn.predict(x=test_pert_images[j:j+1])
            try:
                predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(pert_predictions[0], name="Perturbed", cls=CLS)
                if count > 0:
                    plot_image_with_boxes(img=np.ascontiguousarray(test_pert_images[j], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class)
                    plt.savefig(DIR+"pert_test_image_{}.png".format(j))
                    # print('\n')
            except Exception as e:
                pass

            for k in range(len(test_labels['boxes'][j])):
                cnt += 1
                patched_images = attack.apply_patch(x=test_images[j:j+1], patch_external=patch, gts_boxes=test_gts_boxes[j][k:k+1], random_location=True)
                # collect loss
                loss = get_loss(frcnn, copy.deepcopy(patched_images), y_target)
                loss_history = append_loss_history(loss_history, loss, 1) # will normalize at the end with cnt
 
                patch_predictions = frcnn.predict(x=patched_images)
                
                try:
                    predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(patch_predictions[0], name="Patched", cls=CLS)
                    if count > 0:
                        plot_image_with_boxes(img=np.ascontiguousarray(patched_images[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=predictions_class)
                        plt.savefig(DIR+"patched_test_image_{}_{}.png".format(j,k))
                except Exception as e:
                    pass

        print("patched loss:", loss_history, cnt)
        print("original loss:", original_loss_history)

        with open(DIR+"loss_history_{}.json".format(eps), "a") as file:
            file.write(json.dumps(loss_history))
            file.write(json.dumps(original_loss_history))

        np.save(os.path.join(DIR+"patch_{}".format(eps)), patch)

if __name__ == "__main__":
    main()

