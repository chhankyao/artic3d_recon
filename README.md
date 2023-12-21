# ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections (NeurIPS 2023)
### [Project Page](https://chhankyao.github.io/artic3d/) | [Video](https://youtu.be/r1uKgqWlfyY) | [Paper](https://arxiv.org/abs/2306.04619)

This repository contains a re-implementation of the code for the paper "[ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections](https://chhankyao.github.io/artic3d/)" (NeurIPS 2023). A diffusion-guided optimization framework to estimate the 3D shape and texture of articulated animal bodies from sparse and noisy image in-the-wild.

[Chun-Han Yao](https://www.chhankyao.com/)<sup>1</sup>, [Amit Raj](https://amitraj93.github.io/)<sup>2</sup>, [Wei-Chih Hung](https://hfslyc.github.io/)<sup>3</sup>, [Yuanzhen Li](http://people.csail.mit.edu/yzli/)<sup>2</sup>, [Michael Rubinstein](http://people.csail.mit.edu/mrub/)<sup>2</sup>, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<sup>124</sup><br>, [Varun Jampani](https://varunjampani.github.io)<sup>2</sup><br>
<sup>1</sup>UC Merced, <sup>2</sup>Google Research, <sup>3</sup>Waymo, <sup>4</sup>Yonsei University

![](figures/teaser.png)


## Setup

This repo is largely based on [LASSIE](https://github.com/google/lassie). A python virtual environment is used for dependency management. The code is tested with Python 3.7, PyTorch 1.11.0, CUDA 11.3. First, to install PyTorch in the virtual environment, run:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then, install other required packages by running:

```
pip install -r requirements.txt
```


## Data preparation

### E-LASSIE web images (zebra, giraffe, tiger, elephant, kangaroo, penguin)
* Download LASSIE images following [here](https://github.com/google/lassie) and place them in `data/lassie/images/`.
* Download LASSIE annotations following [here](https://github.com/google/lassie) and place them in `data/lassie/annotations/`.
* Download E-LASSIE (occluded) images from [here](https://www.dropbox.com/s/d9tdrrm7joajfqe/images_occ.zip?dl=0) and place them in `data/lassie/images/`.
* Download E-LASSIE (occluded) annotations from [here](https://www.dropbox.com/s/rzgd1dxzl30rqug/annotations_occ.zip?dl=0) and place them in `data/lassie/annotations/`.
* Preprocess images and extract DINO features of an animal class (e.g. zebra) by running:
```
python preprocess_lassie.py --cls zebra
```
To accelerate feature clustering, try setting number of threads lower (e.g. `OMP_NUM_THREADS=4`).


### Pascal-part (horse, cow, sheep)
* Download Pascal images [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/#devkit) and place them in `data/pascal_part/JPEGImages/`.
* Download Pascal-part annotations [here](http://roozbehm.info/pascal-parts/pascal-parts.html) and place them in `data/pascal_part/Annotations_Part/`.
* Download Pascal-part image sets following [here](https://github.com/google/lassie) and place them in `data/pascal_part/image-sets/`.
* Preprocess images and extract DINO features of an animal class (e.g. horse) by running:
```
python preprocess_pascal.py --cls horse
```


## Skeleton extraction

After preprocessing the input data, apply Hi-LASSIE to extract a 3D skeleton from a specified reference image in the ensemble. For instance, run the following to use the 5-th instance as reference:
```
python extract_skeleton.py --cls zebra --idx 5
```

Following Hi-LASSIE, we recommend selecting an instance where most body parts are visible (e.g. clear side-view). One can also see the DINO feature clustering results in `results/zebra/` to select a good reference or try running the optimization with different skeletons.


## ARTIC3D optimization

To run ARTIC3D optimization on all images in an ensemble jointly, first run:

```
python train.py --cls zebra
```

After the joint optimization, we perform diffusion-guided optimization on a particular instance (e.g. 0) by running:
```
python train.py --cls zebra --inst True --idx 0
```

The qualitative results can be found in `results/zebra/`. The optimization settings can be changed in `main/config.py`. For instance, one can reduce the rendering resolution by setting `input_size` if out of memory.


## Animation fine-tuning

Finally, we can generate 2D animation by 3D part transformations and 2D fine-tuning with our T-DASS module. To obtain the animation of zebra instance 0, run:

```
python animate.py --cls zebra --inst True --idx 0
```

The results can be found in `results/zebra/`.


## Evaluation

Once optimization on all instances is completed, quantitative evaluation can be done by running:

```
python eval.py --cls zebra
```

For the animal classes in E-LASSIE dataset, we report the keypoint transfer accuracy (PCK). For Pascal-part animals, we further calculate the 2D IoU against ground-truth masks.


## Citation

```
@inproceedings{yao2023artic3d,
  title         = {ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections},
  author        = {Yao, Chun-Han and Raj, Amit and Hung, Wei-Chih and Li, Yuanzhen and Rubinstein, Michael and Yang, Ming-Hsuan and Jampani, Varun},
  journal       = {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2023},
}
```
