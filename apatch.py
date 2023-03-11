# MIT License: 
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

import torchvision 
import torchvision.transforms.functional as TF
import os
import cv2
import copy
from PIL import Image

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
import warnings

import numpy as np
from tqdm import tqdm
import logging
import math
import pdb
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin


if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)

# -*- coding: utf-8 -*-

warnings.simplefilter("ignore", UserWarning)

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[0, top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[1] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h][np.newaxis, ...]

def plasma_fractal(mapsize=512, wibbledecay=3):
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

def load_normalized_bases():
    images = os.listdir('frost')
    frost_bases = []
    for im in images:
        image = cv2.imread('frost/' + im)[..., [2, 1, 0]]
        image = np.expand_dims(image, 0)
        frost_bases.append(image)
    return frost_bases

bases = load_normalized_bases()

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

class AngelicPatch(EvasionAttack):
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
        estimator2: "OBJECT_DETECTOR_TYPE" = None,
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        verbose: bool = True,
        im_length: int = 224,
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
        self.estimator2 = estimator2
        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.im_length = im_length
        self.cdict = ["frost", "brightness", "fog", "gaussian_noise", "shot_noise", "impulse_noise", 
                        "speckle_noise", "defocus_blur", "motion_blur", "zoom_blur", "snow", "jpeg_compression", 
                        "pixelate", "contrast", "glass_blur", "spatter"]
        self._check_params()

        self.target_label: Optional[Union[int, np.ndarray, List[int]]] = list()

    def generate(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        target_label: Optional[Union[int, List[int], np.ndarray]] = None,
        aware=True,
        model_name="frcnn",
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
        
        for i_image in range(x.shape[0]):
            target_dict = dict()
            target_dict['boxes'] = labels['boxes'][i_image]
            target_dict["labels"] = labels['labels'][i_image]
            target_dict["scores"] = 1.0 * np.ones([len(labels['labels'][i_image])])

            patch_target.append(target_dict)

        # initialize patch
        patch = Variable((torch.ones(1, 3, 16, 16)*125), requires_grad=True).cuda()
        x = torch.Tensor(x).cuda()
        
        acc_loss1, acc_loss2 = [], []
        for i_step in range(self.max_iter):
            num_batches = math.ceil(x.shape[0] / self.batch_size)
            index = np.arange(num_batches)
            np.random.shuffle(index)
            if i_step % 1 == 0: 
                train_loss_final = [0, 0]

            for i_batch in tqdm(index):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = (i_batch + 1) * self.batch_size
                batch_images = x[i_batch_start:i_batch_end]
                batch_target = patch_target[i_batch_start]
                batch_target_cuda = {}
                batch_target_cuda['boxes']=torch.Tensor(batch_target['boxes']).cuda()
                batch_target_cuda['scores']=torch.Tensor(batch_target['scores']).cuda()
                batch_target_cuda['labels']=torch.Tensor(batch_target['labels']).cuda().to(torch.int64)

                patched_images, _ = self.random_patch_image(copy.deepcopy(batch_images), patch, batch_target_cuda['boxes'])
                if aware:
                    # corruption-aware level 3 training
                    patched_images = self.frost(patched_images, bases, train=aware, severity=3)
                patched_images = patched_images.to(torch.float32) / 255.
                self.estimator._model.train()
                loss = self.estimator._model(patched_images, [batch_target_cuda])
                if model_name == "frcnn":
                    loss_sum = loss['loss_box_reg'] + loss['loss_rpn_box_reg'] + 0.1* loss['loss_classifier']
                else:
                    loss_sum = loss['bbox_regression'] + 0.05*loss['classification'] # ssd
                loss_sum.retain_grad()
                grad = torch.autograd.grad(loss_sum, patch)[0]
                updated_patch = fgsm_attack(patch, self.learning_rate, grad)
                loss_sum.grad.zero_()
                self.estimator._model.zero_grad() 
                grad.zero_()
                patch = torch.clamp(updated_patch,0,255)

                if i_step % 1 == 0:  
                    if model_name == "frcnn":
                        train_loss_final[0] += loss['loss_box_reg'].data / len(index)
                        train_loss_final[1] += loss['loss_classifier'].data / len(index)
                    else:
                        train_loss_final[0] += loss['bbox_regression'].data / len(index)
                        train_loss_final[1] += loss['classification'].data / len(index)
            
            acc_loss1.append(train_loss_final[0].cpu().numpy())
            acc_loss2.append(train_loss_final[1].cpu().numpy())
            print("regression_loss:", train_loss_final[0].item(), "classification_loss:", train_loss_final[1].item())
        return patch.detach().cpu().numpy(), acc_loss1, acc_loss2


    def generate_cross(  # pylint: disable=W0221
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

                patched_images, _ = self.random_patch_image(copy.deepcopy(batch_images), patch, batch_target_cuda['boxes'])
                patched_images = self.frost(patched_images, bases, train=True, severity=3)
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
            print("ssd, regression_loss:", train_loss_final[0].item(), "classification_loss:", train_loss_final[1].item())
            print("frcnn, regression_loss:", train_loss_final2[0].item(), "classification_loss:", train_loss_final2[1].item())
        return patch.detach().cpu().numpy(), acc_loss1, acc_loss2

    def center_box_position(self, bbox, mask_length):
        # converted bbox, start points end points
        bbox = bbox.cpu().numpy().astype(int)
        x00 = max((bbox[2]+bbox[0] - mask_length)//2, 0)
        y00 = max((bbox[3]+bbox[1] - mask_length)//2, 0)

        x0 = min(x00, self.im_length-mask_length)
        y0 = min(y00, self.im_length-mask_length)
        return (y0, y0 + mask_length, x0, x0 + mask_length)

    def random_box_position(self, bbox, mask_length):
        # converted bbox, start points end points
        bbox = bbox.cpu().numpy().astype(int)
        x00 = max((bbox[2]+bbox[0] - mask_length)//2, 0)
        y00 = max((bbox[3]+bbox[1] - mask_length)//2, 0)

        bias_x = np.random.randint(x00 - bbox[0])
        sign = np.random.randint(1)
        x00 += int(bias_x * 2 * (sign - 0.5))
        bias_y = np.random.randint(y00 - bbox[1])
        sign = np.random.randint(1)
        y00 += int(bias_y * 2 * (sign - 0.5))

        x0 = min(x00, self.im_length-mask_length)
        y0 = min(y00, self.im_length-mask_length)
        return (y0, y0 + mask_length, x0, x0 + mask_length)

    def random_patch_image(self, img, patch, bboxs, partial=False, randplace=False):
        if partial:
            rand_n = max(np.random.randint(bboxs.shape[0]), 1)
            add_ns = np.random.choice(bboxs.shape[0], rand_n, replace=False)
        else:
            rand_n = bboxs.shape[0]
            add_ns = range(bboxs.shape[0])
        for i in add_ns:
            box_width = bboxs[i,2]-bboxs[i,0] 
            box_height = bboxs[i,3]-bboxs[i,1]
            height, width = patch.shape[2], patch.shape[3]
            n = np.random.randint(0, 3)
            ratio = min(0.5 * min(box_width, box_height)/width, 5)  # bounded
            patch1 = rot_img(patch, np.pi/2, dtype=torch.cuda.FloatTensor)
            m = nn.UpsamplingBilinear2d(scale_factor=ratio)
            patch1 = m(patch1)
            rl = patch1.shape[-1]
            if randplace:
                coors = self.random_box_position(bboxs[i], rl) 
            else:
                coors = self.center_box_position(bboxs[i], rl) 
            img[:,:,coors[0]:coors[1], coors[2]:coors[3]] = patch1
        return img, rand_n

    def gaussian_noise(self, x, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def shot_noise(self, x, severity=1):
        c = [60, 25, 12, 5, 3][severity - 1]

        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255

    def impulse_noise(self, x, severity=1):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255

    def speckle_noise(self, x, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def glass_blur(self, x, severity=1):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(self.im_length - c[1], c[1], -1):
                for w in range(self.im_length - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[:, h, w], x[:, h_prime, w_prime] = x[:, h_prime, w_prime], x[:, h, w]

        return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255

    def defocus_blur(self, x, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 3, 0))  # 3x224x224 -> 224x224x3

        return np.clip(channels, 0, 1) * 255

    def motion_blur(self, x, severity=1):
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        output = BytesIO()
        x = PILImage.fromarray(np.uint8(x[0]))
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                        cv2.IMREAD_UNCHANGED)

        if x.shape != (self.im_length, self.im_length):
            return np.expand_dims(np.clip(x[..., [2, 1, 0]], 0, 255), 0)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


    def zoom_blur(self, x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return np.clip(x, 0, 1) * 255

    def snow(self, x, severity=1):
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        x = np.array(x, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[1:3], loc=c[0], scale=c[1])[np.newaxis, ...]   # [:2] for monochrome

        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x[0], cv2.COLOR_RGB2GRAY).reshape(self.im_length, self.im_length, 1) * 1.5 + 0.5)
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

    def spatter(self, x, severity=3):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
            (0.65, 0.3, 3, 0.68, 0.6, 0),
            (0.65, 0.3, 2, 0.68, 0.5, 0),
            (0.65, 0.3, 1, 0.65, 1.5, 1),
            (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=3)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255

    def jpeg_compression(self, x, severity=1):
        c = [25, 18, 15, 10, 7][severity - 1]

        output = BytesIO()
        x = PILImage.fromarray(np.uint8(x[0]))
        x.save(output, 'JPEG', quality=c)
        
        x = PILImage.open(output)

        return np.expand_dims(np.array(x), 0)

    def pixelate(self, x, severity=1):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        x = PILImage.fromarray(np.uint8(x[0]))

        x = x.resize((int(self.im_length * c), int(self.im_length * c)), PILImage.BOX)
        x = x.resize((self.im_length, self.im_length), PILImage.BOX)

        return np.expand_dims(np.array(x), 0)

    # /////////////// Corruptions ///////////////
    def frost(self, x, bases, train=False, severity=1):
        c = [(1, 0.4),
            (0.8, 0.6),
            (0.7, 0.7),
            (0.65, 0.7),
            (0.6, 0.75)][severity - 1]
        idx = np.random.randint(len(bases))
        frost = bases[idx]
        x_start, y_start = np.random.randint(0, frost.shape[1] - self.im_length), np.random.randint(0, frost.shape[2] - self.im_length)
        frost_piece = frost[:, x_start:x_start + self.im_length, y_start:y_start + self.im_length, :]
        if train:
            frost_piece = np.transpose(frost_piece, (0,3,1,2))
            return torch.clamp(c[0] * x + c[1] * torch.Tensor(frost_piece).cuda(), 0, 255)
        # if coords is None:
        else:
            return np.clip(c[0] * x + c[1] * frost_piece, 0, 255)
        # else:
        #     # training
        #     patch_frosts = []
        #     for i in range(len(coords)):
        #         patch_frosts.append(frost_piece[:, coords[i][0]:coords[i][1], coords[i][2]:coords[i][3], :])
        #     return np.clip(c[0] * x + c[1] * frost_piece, 0, 255), patch_frosts

    def contrast(self, x, severity=1):
        c = [0.4, .3, .2, .1, .05][severity - 1]
        x1 = np.reshape(x, [1,-1,3])
        means = np.mean(x1, axis=1)
        return np.clip((x - means) * c + means, 0, 255)

    def brightness(self, x, coords=None, severity=1):  #(w, h, c) # 3 224 224
        c = [.1, .2, .3, .4, .5][severity - 1]
        x = np.clip((1 + c) * x, 0, 255)
        return x

    def fog(self, x, coords=None, severity=1):
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
        change = c[0] * plasma_fractal(wibbledecay=c[1])[:self.im_length, :self.im_length][..., np.newaxis]
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

    def apply_multi_affine(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        corrupt_type: int = 0,
        random_location: bool = False,
        masks=[],
        clear=False,
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
        patched_images, _ = random_patch_image(copy.deepcopy(x), copy.deepcopy(patch_external), gts_boxes)
        perspective_transformer = torchvision.transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0)    #.RandomPerspective(distortion_scale=np.random.rand()*0.5+0.2, p=1, interpolation=Image.NEAREST)
        patched_images = Image.fromarray(np.uint8(patched_images[0].permute(1,2,0).cpu().numpy())).convert('RGB')
        patched_images, ret = perspective_transformer(patched_images)
        transferred_masks = [TF.affine(masks[i],  *ret) for i in range(len(masks))]
        new_bbox = []

        # 3, 300, 300, background should be 0
        for ii in range(len(transferred_masks)):
            obj_x, obj_y = np.where(transferred_masks[ii][0] > 0.5)
            if len(obj_x) > 0 and len(obj_y) > 0:
                min_x, max_x = np.min(obj_x), np.max(obj_x)
                min_y, max_y = np.min(obj_y), np.max(obj_y)
                new_bbox.append([min_y, min_x, max_y, max_x])
            else:
                new_bbox.append([0, 0, self.im_length, self.im_length])
        
        patched_images = np.expand_dims(np.array(patched_images).astype(np.float32), 0)
        if not clear:
            patched_images = self.frost(patched_images, bases, severity=3)
        
        return patched_images.astype(np.float32), np.array(new_bbox), transferred_masks

    def apply_no_patch(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        corrupt_type: int = 0,
        random_location: bool = False,
        masks=[],
        clear=False,
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
        perspective_transformer = torchvision.transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0)  
        patched_images = Image.fromarray(np.uint8(x[0].permute(1,2,0).cpu().numpy())).convert('RGB')
        patched_images, ret = perspective_transformer(patched_images)
        transferred_masks = [TF.affine(masks[i],  *ret) for i in range(len(masks))]
        new_bbox = []

        # 3, 300, 300, background should be 0
        for ii in range(len(transferred_masks)):
            obj_x, obj_y = np.where(transferred_masks[ii][0] > 0.5)
            if len(obj_x) > 0 and len(obj_y) > 0:
                min_x, max_x = np.min(obj_x), np.max(obj_x)
                min_y, max_y = np.min(obj_y), np.max(obj_y)
                # normalized_x, normalized_y, normalized_x+normalized_width, normalized_y+normalized_height
                new_bbox.append([min_y, min_x, max_y, max_x])
            else:
                new_bbox.append([0, 0, self.im_length, self.im_length])

        patched_images = np.expand_dims(np.array(patched_images).astype(np.float32), 0)
        if not clear:
            patched_images = self.frost(patched_images, bases, severity=3)
        return patched_images.astype(np.float32), np.array(new_bbox)


    def apply_multi_patch(
        self,
        x: np.ndarray,
        gts_boxes,
        patch_external: Optional[np.ndarray] = None,
        corrupt_type: int = 0,
        severity = 1,
        clear = False,
        rp = False,
        partial=False,
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
        # torch.Size([1, 3, 224, 224])
        patched_images, rand_n = self.random_patch_image(copy.deepcopy(x), 
                                            copy.deepcopy(patch_external), 
                                            gts_boxes, 
                                            randplace=rp,
                                            partial=partial)
        # torch.Size([1, 3, 224, 224])
        patched_images = patched_images.permute(0,2,3,1).cpu().numpy()
        # patched_images should be a tensor (0, 255)
        if not clear:
            if corrupt_type < 1:
                patched_images = getattr(self, self.cdict[corrupt_type])(patched_images, bases, severity=severity)
            else:
                patched_images = getattr(self, self.cdict[corrupt_type])(patched_images, severity=severity)
        # (1, 224, 224, 3)
        return patched_images, rand_n

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
