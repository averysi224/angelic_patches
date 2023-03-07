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
""" global ssd multiple
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
from torch.autograd import Variable

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
im_length = 224


def frost(x, bases, coords=None, severity=3):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(len(bases))
    frost = bases[idx]
    # x shape = torch.Size([1, 3, 300, 300])
    x_start, y_start = np.random.randint(0, frost.shape[1] - im_length), np.random.randint(0, frost.shape[2] - im_length)
    frost_piece = frost[:, x_start:x_start + im_length, y_start:y_start + im_length, :] 
    frost_piece = np.transpose(frost_piece, (0,3,1,2))
    if coords is None:
        return torch.clamp(c[0] * x + c[1] * torch.Tensor(frost_piece).cuda(), 0, 255)
    else:
        patch_frosts = []
        for i in range(len(coords)):
            patch_frosts.append(frost_piece[:, :, coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]])
        return torch.clamp(c[0] * x + c[1] * torch.Tensor(frost_piece).cuda(), 0, 255)


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
    x = torch.clamp((1 + c) * x, 0, 255)
    
    return x, 1+c

def fog(x, coords=None, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    change = c[0] * plasma_fractal(wibbledecay=c[1])[:im_length, :im_length][..., np.newaxis]
    change = np.transpose(np.repeat(change, [3], axis=2) * 255, [2,0,1])
    max_val = x.max() / 255.
    xx = x + torch.Tensor(change).cuda()
    xx = torch.clamp(xx * max_val / (max_val + c[0]), 0, 255) 
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

bases = load_normalized_bases()

def center_box_position(bbox, mask_length):
    # converted bbox, start points end points
    bbox = bbox.cpu().numpy().astype(int)
    x00 = max((bbox[2]+bbox[0] - mask_length)//2, 0)
    y00 = max((bbox[3]+bbox[1] - mask_length)//2, 0)

    x0 = min(x00, im_length-mask_length)
    y0 = min(y00, im_length-mask_length)
    return (y0, y0 + mask_length, x0, x0 + mask_length)

def random_patch_image(img, patch, bboxs, test=False):
    for i in range(bboxs.shape[0]):
        box_width = bboxs[i,2]-bboxs[i,0] 
        box_height = bboxs[i,3]-bboxs[i,1]
        height, width = patch.shape[2], patch.shape[3]
        n = np.random.randint(0, 3)
        ratio = min(0.5*min(box_width, box_height)/width, 5)  # bounded
        patch1 = rot_img(patch, np.pi/2, dtype=torch.cuda.FloatTensor)
        m = nn.UpsamplingBilinear2d(scale_factor=ratio)
        patch1 = m(patch1)
        rl = patch1.shape[-1]
        coors = center_box_position(bboxs[i], rl) # looks fine
        img[:,:,coors[0]:coors[1], coors[2]:coors[3]] = patch1
    return img

def random_patch_image_perspect(img, patch, bboxs, test=False):
    for i in range(bboxs.shape[0]):
        box_width = bboxs[i,2]-bboxs[i,0] 
        box_height = bboxs[i,3]-bboxs[i,1]
        height, width = patch.shape[2], patch.shape[3]
        ratio = min(0.5*min(box_width, box_height)/width, 5)  # bounded
        patch1 = rot_img(patch, np.pi/2, dtype=torch.cuda.FloatTensor)
        m = nn.UpsamplingBilinear2d(scale_factor=ratio)
        patch1 = m(patch1)
        mask = (patch1[:, 0] < 10) & (patch1[:, 1] < 10) & (patch1[:, 2] < 10)
        rl = patch1.shape[-1]
        coors = center_box_position(bboxs[i], rl) # looks fine
        img[:,:,coors[0]:coors[1], coors[2]:coors[3]] = img[:,:,coors[0]:coors[1], coors[2]:coors[3]] * mask + patch1 * ~mask
    return img

def fgsm_attack(patch, epsilon, data_grad):
    updated_patch = patch - epsilon * torch.sign(data_grad) #/255.
    updated_patch = torch.clamp(updated_patch, 0, 255)
    return updated_patch

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    # x.size is a int for np array
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

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
        estimator2: "OBJECT_DETECTOR_TYPE",
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
        self.estimator2 = estimator2

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
        patch_target: List[Dict[str, np.ndarray]] = list()

        # predictions = self.estimator.predict(x=patched_images, standardise_output=True)
        labels = kwargs.get("labels")
        
        # one image whole target
        for i_image in range(x.shape[0]):
            target_dict = {}
            target_dict['boxes']=torch.Tensor(labels['boxes'][i_image]).cuda()
            target_dict['scores']=torch.Tensor(1.0 * np.ones([len(labels['labels'][i_image])])).cuda()
            target_dict['labels']=torch.Tensor(labels['labels'][i_image]).cuda().to(torch.int64)

            patch_target.append(target_dict)

        # initialize patch
        patch = Variable((torch.ones(1, 3, 16, 16)*125), requires_grad=True).cuda()
        x = torch.Tensor(x).cuda()
        
        acc_loss1, acc_loss2 = [], []
        for i_step in range(self.max_iter):
            num_batches = math.ceil(x.shape[0] / self.batch_size)
            index = np.arange(num_batches)
            np.random.shuffle(index)
            if i_step % 1 == 0: #== self.max_iter - 1:
                train_loss_final = [0, 0]
                train_loss_final2 = [0, 0]

            for i_batch in tqdm(index):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = (i_batch + 1) * self.batch_size
                batch_images = x[i_batch_start:i_batch_end]
                batch_target = patch_target[i_batch_start]
                batch_target_cuda = {}

                batch_target_cuda['boxes']=batch_target['boxes']
                batch_target_cuda['scores']=batch_target['scores']
                batch_target_cuda['labels']=batch_target['labels']

                patched_images = random_patch_image(copy.deepcopy(batch_images), patch, batch_target_cuda['boxes'])
                patched_images = frost(patched_images, bases)
                patched_images = patched_images.to(torch.float32) / 255. 
                # should put image list inside, however as its single image, its the same
                self.estimator._model.train()
                loss = self.estimator._model(patched_images, [batch_target_cuda])
                self.estimator2._model.train()
                loss2 = self.estimator2._model(patched_images, [batch_target_cuda])
                loss_sum = loss['bbox_regression'] + 0.1*loss['classification'] # ssd
                loss_sum.retain_grad()
                grad = torch.autograd.grad(loss_sum, patch, retain_graph=True)[0]
                
                updated_patch = fgsm_attack(patch, self.learning_rate, grad)
                loss_sum.grad.zero_()
                self.estimator._model.zero_grad() 
                grad.zero_()

                # frcnn
                
                loss_sum2 = loss2['loss_box_reg'] + loss2['loss_rpn_box_reg'] + 0.1* loss2['loss_classifier']
                loss_sum2.retain_grad()
                grad2 = torch.autograd.grad(loss_sum2, patch)[0]

                updated_patch = fgsm_attack(updated_patch, self.learning_rate, grad2)

                loss_sum2.grad.zero_()
                self.estimator2._model.zero_grad() 
                grad2.zero_()
                patch = torch.clamp(updated_patch,0,255)

                if i_step % 1 == 0:  # == self.max_iter - 1:
                    train_loss_final[0] += loss['bbox_regression'].data / len(index)
                    train_loss_final[1] += loss['classification'].data / len(index)

                    train_loss_final2[0] += loss2['loss_box_reg'].data / len(index)
                    train_loss_final2[1] += loss2['loss_classifier'].data / len(index)
            
            acc_loss1.append(train_loss_final[0].cpu().numpy())
            acc_loss2.append(train_loss_final[1].cpu().numpy())
            print("ssd:", train_loss_final)
            print("frcnn:", train_loss_final2)
        return patch.detach().cpu().numpy(), acc_loss1, acc_loss2

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
        patched_images = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)
        patched_images = frost(patched_images, bases)
        # patched_images = contrast(patched_images)

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
        patched_images = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)

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
        # perspective_transformer = torchvision.transforms.RandomPerspective(distortion_scale=np.random.rand()*0.5+0.2, p=1, interpolation=Image.NEAREST)
        # patch = torchvision.transforms.ToPILImage()(patch_external[0].cpu())
        # patch = np.expand_dims(np.array(perspective_transformer(patch)), 0)
        # patch = torch.Tensor(patch).cuda().permute([0,3,1,2])
        # patched_images = random_patch_image_perspect(copy.deepcopy(x), patch, gts_boxes, True)
        patched_images = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes, True)
        patched_images = frost(patched_images, bases)
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
