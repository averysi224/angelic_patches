import cv2
import numpy as np
import matplotlib.pyplot as plt

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DPatch, RobustDPatch
import torch
import dsld
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

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

def comp_mask(image, positions, labels, cls=1):
    length = np.sum(labels == cls)
    mask = np.zeros([image.shape[0], image.shape[1]])
    if length == 0:
        mask[int(image.shape[0]/2), int(image.shape[1]/2)] = 1
        return mask
    else:
        positions = np.array(positions)[labels==cls, :, :].reshape(-1, 4)
        center_x, center_y = (positions[:, 0] + positions[:, 2]) / 2.0, (positions[:, 1] + positions[:, 3]) / 2.0
        mask[center_y.astype(int), center_x.astype(int)] = 1
    return mask

def extract_predictions(predictions_, cls=1):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(predictions_["boxes"].astype(int))]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.1
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]
    predictions_["labels"] = predictions_["labels"][: predictions_t + 1]

    ## Count how many objects' score is higher than threshold
    count = predictions_["labels"][predictions_["labels"] == cls].shape[0]

    # return predictions_class, predictions_boxes, predictions_score
    return predictions_["labels"], predictions_boxes, predictions_score, count


# def plot_image_with_boxes(img, boxes, pred_cls):
#     text_size = 5
#     text_th = 5
#     rect_th = 6

#     for i in range(len(boxes)):
#         # Draw Rectangle with the coordinates
#         cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)

#         # Write the prediction class
#         cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

#     plt.axis("off")
#     plt.imshow(img.astype(np.uint8), interpolation="nearest")
#     plt.show()

def plot_image_with_boxes(img, boxes, pred_cls, cls=1):
    text_size = 1
    text_th = 1
    rect_th = 1

    for i in range(len(boxes)):
        if pred_cls[i] == cls:
            # Draw Rectangle with the coordinates
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(255, 255, 255), thickness=rect_th)

            # breakpoint()
            # Write the prediction class
            cv2.putText(img, str(pred_cls[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), thickness=text_th)

    plt.axis("off")
    img = img[:, :, ::-1]
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    # plt.savefig('test.png')
    # plt.show()


def main():
    # Create ART object detector
    # frcnn = PyTorchFasterRCNN(
    #     clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    # )
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier"]
    )

    path2data = '/data5/wenwens/coco2017/train2017'
    path2json = '/data5/wenwens/coco2017/annotations/instances_train2017.json'

    # create own Dataset
    my_dataset = dsld.myOwnDataset(root=path2data,
                            annotation=path2json,
                            transforms=dsld.get_transform()
                            )

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Batch size
    train_batch_size = 5

    # my_dataset = transforms.ToPILImage()(my_dataset)
    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    masks = []
    image = []
    gts_boxes = []
    gts_labels = []
    # DataLoader is iterable over Dataset
    for idx, [imgs, annotations] in enumerate(data_loader):
        if idx > 1:
            break
        imgs = torch.stack(imgs)
        imgs = imgs.permute(0,2,3,1)
        imgs = imgs.numpy()[:, :,:, ::-1]*255.0
        # image.append(imgs)
        predictions = frcnn.predict(x=imgs)
        for i in range(imgs.shape[0]):
            print("\nPredictions image {}:".format(i+train_batch_size*idx))
            # Process predictions
            predictions_class, predictions_boxes, predictions_scores, count = extract_predictions(predictions[i], cls=1)

            # Plot predictions
            gt_class, gt_boxes = annotations[i]['labels'].numpy(), annotations[i]['boxes'].numpy().astype(int)
            count = np.sum(gt_class == 1)
            print('count', count)

            boxes = []
            for j in range(gt_boxes.shape[0]):
                boxes.append([(gt_boxes[j][0], gt_boxes[j][1]), (gt_boxes[j][2], gt_boxes[j][3])])
            
            if count > 0:
                image.append(imgs[i])
                plot_image_with_boxes(img=imgs[i].copy(), boxes=predictions_boxes, pred_cls=predictions_class)
                mask = comp_mask(imgs[i], boxes, gt_class, cls=1)
                masks.append(mask)
                gts_boxes.append(gt_boxes)
                gts_labels.append(gt_class)

                plt.savefig(f'original detection {i+train_batch_size*idx}.png')
            # breakpoint()
    masks = np.stack(masks).astype(bool)
    image = np.stack(image)

    attack = DPatch(
            frcnn,
            patch_shape=(5, 5, 3),
            learning_rate=2.0,
            max_iter=300,
            batch_size=5,
            verbose=False,
        )
    patch = attack.generate(x=image, target_label=1, mask=masks, labels={'boxes':gts_boxes, 'labels':gts_labels})
    plt.axis("off")
    plt.title("Adversarial Patch")
    plt.imshow(patch.astype(np.uint8), interpolation="nearest")
    plt.savefig("adversarial patch.png")
    patched_images = attack.apply_patch(x=image, patch_external=patch, random_location=True, mask=masks)
    # patched_images = attack.apply_patch(x=image, patch_external=patch, random_location=False, mask=masks)
    # breakpoint()
    # print("\nThe attack budget eps is {}".format(eps))
    # print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(image - image_adv))))

    for i in range(patched_images.shape[0]):
        plt.axis("off")
        plt.title("patched image {}".format(i))
        plt.imshow(patched_images[i].astype(np.uint8)[:, :, ::-1], interpolation="nearest")
        plt.show()
        plt.savefig("patcheded image {}.png".format(i))

    predictions_adv = frcnn.predict(x=patched_images)
    # breakpoint()

    for i in range(patched_images.shape[0]):
        print("\nPredictions adversarial image {}:".format(i))

        # Process predictions
        predictions_adv_class, predictions_adv_boxes, predictions_adv_predictions_scores, count = extract_predictions(predictions_adv[i])
        print('count', count)

        # Plot predictions
        plot_image_with_boxes(img=patched_images[i].copy(), boxes=predictions_adv_boxes, pred_cls=predictions_adv_class)

        plt.savefig(f'attacked detection {i}.png')


if __name__ == "__main__":
    main()
