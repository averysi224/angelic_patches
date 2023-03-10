Here we provide the Faster-RCNN agnostic patch code.
We build our code based on the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) 
For installation guidelines, please refer to README-original.md

## Installation
This code is adapted from the adversarial-robustness-toolbox. The mai n contribution code of our paper are in 'art/attacks/evasion/dpatch.py', 'examples/global_frcnn_final.py' and 'python examples/test_frcnn_affine_final.py'.

For environment requirements, please refer to README_original.md. We have also provide a conda environment file in the main folder.

After satisfy the enviroment requirements, you could install the code with command line:
```
cd adversarial-robustness-toolbox
pip install -e .
```

## Preparation
Download the COCO category-wise json files from [Google Drive:](https://drive.google.com/file/d/1rJLqXY4tUAGGjG82stwHoCapfSTf3p_y/view?usp=share_link)
unzip the compressed folder in main folder and name it: category_json.

In the affine robustness experiments, we need to extract the transformation parameters of the torchvision RandomAffine transformation. Thus please open your torchvision install directory and edit the forward function in the RandomAffine class, e.g. '/home/xxx/anaconda3/envs/myclone/lib/python3.7/site-packages/torchvision/transforms/transforms.py'. 

edit the return from 

```
return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
```

to 

```
return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center), ret
```

## Example Testing
We provide example patches for all baseline testing. Each run will compute high confident IoU automatically. Detections of patched and unpatched images are also saved. You can try different categories like "person", "bus", "bottle", "chair", "laptop" etc.

Test F-RCNN affine robustness under frost corruption:
```
python examples/test_affine_final.py --cate bus --coco_path your_COCO_path
```

Test F-RCNN affine robustness without corruption (clear):
```
python examples/test_affine_final.py --cate bus --clear --coco_path your_COCO_path
```

Test F-RCNN corruption-aware patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --coco_path your_COCO_path
```


Test F-RCNN corruption-aware patch robustness without corruption (clear):
```
python examples/global_main.py --cate bus --coco_path your_COCO_path --clear
```

Test F-RCNN corruption-agnostic patch robustness under a series of corruption:
```
python examples/global_main.py --agnostic --cate bus  --coco_path your_COCO_path 
```

Test F-RCNN partially applied (only some of the objects are patched) corruption-aware patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --partial --coco_path your_COCO_path 
```

Test F-RCNN random placed (not in the center) corruption-aware patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --coco_path your_COCO_path --randplace 
```

## Example Training
Train F-RCNN corruption-aware patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --coco_path your_COCO_path --train_patch 
```

Train F-RCNN corruption-agnostic patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --coco_path your_COCO_path --train_patch --agnostic
```
