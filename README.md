## TODO
- check all patch ratios
- fix comments, Variable/Class names
- fix license
- polish argparse and all print functions

## Installation
This code based on the [Adversarial Robustness Toolbox 1.7.2](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/1.7.2) 

- Step 0: For environment requirements, we provide a pip requirement.txt file and a conda environment file in the main folder.

- Step 1: Install adversarial-robustness-toolbox
```
git clone --depth 1 --branch 1.7.2 https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox
pip install .
```
- Step 2: Download our code and run.

## Preparation
### Data Preparation
Download the COCO category-wise json files from [Google Drive:](https://drive.google.com/file/d/1rJLqXY4tUAGGjG82stwHoCapfSTf3p_y/view?usp=share_link)
unzip the compressed folder in main folder and name it: category_json.

### (Optional) Only if you want to run the affine test
In the affine robustness experiments, we need to extract the transformation parameters of the torchvision RandomAffine transformation. Thus please open your torchvision install directory and edit the forward function in the RandomAffine class, e.g. '~/anaconda3/envs/myclone/lib/python3.7/site-packages/torchvision/transforms/transforms.py'. 

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

### Corruption-Aware Tests
Test F-RCNN corruption-aware patch robustness under frost corruption:
```
python angelic_global_main.py --cate bus --coco_path --model_name frcnn your_COCO_path
```

Test F-RCNN corruption-aware patch robustness without corruption (clear):
```
python angelic_global_main.py --cate bus --coco_path --model_name frcnn your_COCO_path --clear
```
### Corruption-Agnostic Tests
Test F-RCNN corruption-agnostic patch robustness under a series of corruption:
```
python angelic_global_main.py --agnostic --cate bus --model_name frcnn --coco_path your_COCO_path 
```
### Corruption-Aware Extra Tests

Test F-RCNN partially applied (only some of the objects are patched) corruption-aware patch robustness under frost corruption:
```
python angelic_global_main.py --cate bus --partial --model_name frcnn --coco_path your_COCO_path 
```

Test F-RCNN random placed (not in the center) corruption-aware patch robustness under frost corruption:
```
python angelic_global_main.py --cate bus --model_name frcnn --coco_path your_COCO_path --randplace 
```

### Corruption-Aware Affine Tests
Test F-RCNN affine robustness under frost corruption:
```
python angelic_affine_main.py --cate bus --model_name frcnn --coco_path your_COCO_path
```

Test F-RCNN affine robustness without corruption (clear):
```
python angelic_affine_main.py --cate bus --model_name frcnn --clear --coco_path your_COCO_path
```

### Corruption-Aware Cross-Model Tests
```
python angelic_affine_main.py --cate bus --model_name retina --coco_path your_COCO_path
```

## Example Training
Train F-RCNN corruption-aware patch robustness under frost corruption:
```
python angelic_global_main.py --cate bus --coco_path your_COCO_path --train_patch 
```

Train F-RCNN corruption-agnostic patch robustness under frost corruption:
```
python examples/global_main.py --cate bus --coco_path your_COCO_path --train_patch --agnostic
```

## Example Cross-Model Training

```
python angelic_affine_main.py --cate bus --model_name retina --train_patch --coco_path your_COCO_path
```