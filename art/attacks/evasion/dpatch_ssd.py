# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" ssd
This module implements the adversarial patch attack `DPatch` for object detectors.
This dpatch implement a training process that 
attach single patch - improve on all instances in the targeted category

| Paper link: https://arxiv.org/abs/1806.02299v4
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

import argparse
import csv
import os
import cv2
import copy
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import argparse

import logging
import math
import random
import pdb
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from scipy import ndimage

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config

import torchvision

if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)

length = 16
im_length = 300

def frost(x, bases, coords=None, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(len(bases))
    frost = bases[idx]
    x_start, y_start = np.random.randint(0, frost.shape[1] - im_length), np.random.randint(0, frost.shape[2] - im_length)
    frost_piece = frost[:, x_start:x_start + im_length, y_start:y_start + im_length, :]
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
    x = (1 + c) * x
    
    return x, 1+c

def fog(x, coords=None, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    change = c[0] * plasma_fractal(wibbledecay=c[1])[:im_length, :im_length][..., np.newaxis]
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

def patch_affine_reverse(patch, n, i_step, i_batch, i):
    fact = length/patch.shape[1]
    patch = ndimage.zoom(patch, [fact,fact,1], mode='nearest')
    patch = ndimage.rotate(patch, -n*6, reshape=False, mode='nearest')
    return patch

def load_normalized_bases():
    images = os.listdir('frost');
    frost_bases = []
    for im in images:
        image = cv2.imread('frost/' + im)[..., [2, 1, 0]]
        image = np.expand_dims(image, 0)
        frost_bases.append(image)
    return frost_bases

bases = load_normalized_bases()

def random_box_position(bbox, mask_length):
    x00, y00 = 0, 0
    # converted bbox, start points end points
    if (bbox[2]-bbox[0]) - mask_length > 0:
        x00 = np.random.randint((bbox[2]-bbox[0]) - mask_length)
    if (bbox[3]-bbox[1]) - mask_length > 0:
        y00 = np.random.randint((bbox[3]-bbox[1]) - mask_length)

    x0 = min(bbox[0] + x00, im_length-mask_length)
    y0 = min(bbox[1] + y00, im_length-mask_length)
    return (y0, y0 + mask_length, x0, x0 + mask_length)

def center_box_position(bbox, mask_length):
    # converted bbox, start points end points
    x00 = max((bbox[2]+bbox[0] - mask_length)//2, 0)
    y00 = max((bbox[3]+bbox[1] - mask_length)//2, 0)

    x0 = min(x00, im_length-mask_length)
    y0 = min(y00, im_length-mask_length)
    return (y0, y0 + mask_length, x0, x0 + mask_length)

def random_patch_image(img, patch, bboxs, test=False):
    coords, masks, angles, circle_rls = [], [], [], []
    for i in range(bboxs.shape[0]):
        # height width is reversed
        patch1, rl, n, mask = patch_rand_affine(patch, bboxs[i,3]-bboxs[i,1], bboxs[i,2]-bboxs[i,0], length)
        coors = center_box_position(bboxs[i], rl) # looks fine
        coords.append(coors)
        masks.append(mask)
        angles.append(n)
        # mask outside 0 inside 1
        img[0,coors[0]:coors[1], coors[2]:coors[3],:] = img[0, coors[0]:coors[1], coors[2]:coors[3],:] * (1-mask) + patch1*mask
        # if test:
        #     predictions_boxes = [[(int(bboxs[i,0]), int(bboxs[i,1])), (int(bboxs[i,2]), int(bboxs[i,3]))]]
        #     plot_image_with_boxes(img=np.ascontiguousarray(img[0], dtype=np.uint8), boxes=predictions_boxes, pred_cls=np.array([6]))
        #     plt.savefig("good/test.png")
   
    return img, coords, masks, angles

def patch_rand_affine(patch, box_height, box_width, mask_length=length):
    height, width = patch.shape[0], patch.shape[1]
    n = np.random.randint(-5, 5)
    ratio = min(max(0.8 * box_width/width, 0.8), 3)
    patch1 = ndimage.zoom(patch, [ratio,ratio,1], mode='nearest')
    rl = patch1.shape[0]
    patch1 = ndimage.rotate(patch1, n*6, reshape=False, mode='nearest')
    xx, yy = np.mgrid[:rl, :rl]
    circle = (xx - rl//2) ** 2 + (yy - rl//2) ** 2
    mask = np.expand_dims(circle < (rl/2)**2, 2)

    return patch1, rl, n, mask

def fgsm_attack(image, epsilon, data_grad, coors, affine_masks):
    perturbed_image = image - epsilon * np.sign(data_grad) / 255.
    perturbed_image = np.clip(perturbed_image, 0, 1)
    return perturbed_image

def get_loss(frcnn, x, y):
    frcnn._model.train()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor_list = list()
    
    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            imgg = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            imgg = transform(x[i]).to(frcnn._device)
        image_tensor_list.append(imgg)

    yy = copy.deepcopy(y[0])
    yy['boxes'] = torch.from_numpy(yy['boxes']).type(torch.float).cuda()
    yy["labels"] = torch.from_numpy(yy['labels']).cuda()
    yy["scores"] = torch.from_numpy(1.0 * np.ones(1)).type(torch.int64).cuda()

    loss = frcnn._model(image_tensor_list, [yy])
    for loss_type in ["bbox_regression", "classification"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss

class DPatch(EvasionAttack):
    """
    Implementation of the DPatch attack.

    | Paper link: https://arxiv.org/abs/1806.02299v4
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.DPatch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial path as a tuple of shape (height, width, nb_channels).
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE)
        else:
            # initialize as gray patch
            self._patch = np.array([125.0, 125.0, 125.0]) * np.ones(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE) / 255 * (self.estimator.clip_values[1] - self.estimator.clip_values[0]) + self.estimator.clip_values[0]
            # no circle
            xx, yy = np.mgrid[:length, :length]
            circle = (xx - length//2) ** 2 + (yy - length//2) ** 2
            self._mask = np.expand_dims((circle < (length/2)**2 ),2)
            # self._patch = patch * self._mask

        self.target_label: Optional[Union[int, np.ndarray, List[int]]] = list()

    def generate(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        target_label: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate DPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :param target_label: The target label of the DPatch attack.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: Adversarial patch.
        """
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is not None:
            raise ValueError("The DPatch attack does not use target labels.")
        if x.ndim != 4:  # pragma: no cover
            raise ValueError("The adversarial patch can only be applied to images.")
        if target_label is not None:
            if isinstance(target_label, int):
                self.target_label = [target_label] * x.shape[0]
            elif isinstance(target_label, np.ndarray):
                if not (  # pragma: no cover
                    target_label.shape == (x.shape[0], 1) or target_label.shape == (x.shape[0],)
                ):
                    raise ValueError("The target_label has to be a 1-dimensional array.")
                self.target_label = target_label.tolist()
            else:
                if not len(target_label) == x.shape[0] or not isinstance(target_label, list):  # pragma: no cover
                    raise ValueError("The target_label as list of integers needs to of length number of images in `x`.")
                self.target_label = target_label

        patch_target: List[Dict[str, np.ndarray]] = list()

        # predictions = self.estimator.predict(x=patched_images, standardise_output=True)
        labels = kwargs.get("labels")
        # # one image single target
        # for i_image in range(x.shape[0]):
        #     per_image_target = []
        #     for i_label in range(len(labels['labels'][i_image])):    
        #         target_dict = dict()
        #         target_dict['boxes'] = labels['boxes'][i_image][i_label:i_label+1]
        #         target_dict["labels"] = labels['labels'][i_image][i_label:i_label+1]
        #         target_dict["scores"] = 1.0 * np.ones([1])
        #         per_image_target.append(target_dict)

        #     patch_target.append(per_image_target)

        # one image whole target
        for i_image in range(x.shape[0]):
            target_dict = dict()
            target_dict['boxes'] = labels['boxes'][i_image]
            target_dict["labels"] = labels['labels'][i_image]
            target_dict["scores"] = 1.0 * np.ones([len(labels['labels'][i_image])])

            patch_target.append(target_dict)

        # initialize patch
        patch = self._patch
        
        for i_step in range(self.max_iter):
            num_batches = math.ceil(x.shape[0] / self.batch_size)
            index = np.arange(num_batches)
            np.random.shuffle(index)

            for i_batch in tqdm(index):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = (i_batch + 1) * self.batch_size
                batch_images = x[i_batch_start:i_batch_end]
                batch_target = patch_target[i_batch_start]

                # for i_label in range(len(batch_target)):
                for i_label in range(batch_target["scores"].shape[0]):
                    if batch_target['labels'][i_label] == target_label:
                        patched_images, coords, masks, angles = random_patch_image(copy.deepcopy(batch_images), copy.deepcopy(patch), batch_target['boxes'][i_label:i_label+1])

                        patched_images, patch_frosts = frost(patched_images, bases, coords)
                        patched_images, patch_pert = contrast(patched_images)
                        # patched_images, change, max_val = fog(patched_images, coords)
                        # patched_images, bc = brightness(patched_images, coords)
                        patched_images = patched_images.astype(np.float32) / 255.
                        # shape (1, 224, 224, 3)
                        # batch_target[0] {'boxes': array([[181,  42, 223, 165]]), 'labels': array([6]), 'scores': array([1.])}
                        gradients = self.estimator.loss_gradient(x=patched_images, y=[batch_target], standardise_output=True,)
                        # set eps to 1 temp
                        perturbed_image = fgsm_attack(patched_images, self.learning_rate, gradients, coords, masks)

                        new_patch = np.zeros_like(self._patch)
                        for i in range(len(coords)):
                            tmp = perturbed_image[0, coords[i][0]:coords[i][1], coords[i][2]:coords[i][3],:].copy() * 255
                            tmp = (tmp - 0.6 * patch_pert[0]) / 0.4  # contrast
                            tmp = tmp - 0.4 * patch_frosts[i] # frost
                            # tmp = tmp / bc  # brightness
                            # tmp = tmp / max_val * (max_val + 1.5) - change  # fog
                            tmp = patch_affine_reverse(tmp[0], angles[i], i_step, i_batch, i)

                            new_patch += tmp

                        new_patch /= len(coords)
                        patch = np.clip(new_patch,0,255)
                    
                    """plt.axis("off")
                    toimg = patched_images[0][:, :, ::-1]
                    plt.imshow(toimg.astype(np.uint8), interpolation="nearest")
                    plt.savefig("good/perturbed_image_{}.png".format(i_step))"""

                # if i_step % 1 == 0:
                #     cnt = 0
                #     problematic = []
                #     loss_history = np.zeros([4])
                #     for i_batch in range(x.shape[0]):
                #         batch_target = patch_target[i_batch]
                #         for i_label in range(len(batch_target)):
                #             # [{'boxes': array([[101,  43, 122,  76]]), 'labels': array([6]), 'scores': array([1.])}]
                #             # pdb.set_trace()
                #             output = get_loss(self.estimator, x[i_batch:i_batch+1].copy(), [batch_target])
                #             for i, loss_type in enumerate(["bbox_regression", "classification"]):
                #                 loss_history[i] += output[loss_type]
                #                 cnt += 1
                #     loss_history /= cnt
                #     print(loss_history)

        return patch

    def apply_single_patch(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        random_location: bool = False,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images.
        """
        patched_images, coords, masks, angles = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)
        patched_images, _ = frost(patched_images, bases, coords)
        # patched_images, _ = contrast(patched_images)
        # patched_images, change, max_val = fog(patched_images, coords)
        # patched_images, bc = brightness(patched_images, coords)
        patched_images = patched_images.astype(np.float32)

        return patched_images

    def apply_no_corruption(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        random_location: bool = False,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images.
        """
        patched_images, coords, masks, angles = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)
        patched_images = patched_images.astype(np.float32)

        return patched_images

    def apply_multi_patch(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        random_location: bool = False,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images.
        """
        patched_images, coords, masks, angles = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)
        patched_images, _ = frost(patched_images, bases, coords)
        # patched_images, _ = contrast(patched_images)
        patched_images = patched_images.astype(np.float32)

        return patched_images

    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(isinstance(s, int) for s in self.patch_shape):
            raise ValueError("The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
