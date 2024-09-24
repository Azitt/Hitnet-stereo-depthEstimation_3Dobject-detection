
# Overview

This repository contains the implementation and evaluation of HitNet ([paper](https://arxiv.org/abs/2007.12140)) for depth estimation and 3D object detection tasks.

![alt text](resources/image.png)

HitNet operates in three main steps:

1- **Feature Extraction**: 
- A U-Net processes the left and right images and extracts features at multiple scales.

2- **Initialization**: 

- Works on each scale of the extracted features. It divides features into 4x4 tiles and tests multiple possible depth values for each tile. Finally, it selects the depth with the lowest error.

3- **Refinement**: 

- Uses results from the initialization step and improves depth predictions using slanted windows. This helps to get more accurate depth estimates.

For more detailed information, please refer to my Medium article [link to your article].

## Inference Results: HitNet vs CREStereo

<div align="center">
  <strong style="display: inline-block; margin: 0 20px;">HitNet disparity map estimation and depth calucation</strong> 
</div>

<div align="center"> 
  <img src="./resources/hit_disparity.gif" alt="Bottom Left Video" width="600"/> 
</div>

<p align="center">
  <strong style="display: inline-block; margin: 0 20px;">CREStereo disparity map estimation and depth calucation</strong>
</p>

<div align="center"> 
  <img src="./resources/cre_disparity.gif" alt="Bottom Right Video" width="600"/> 
</div>

<p align="center">
  <strong style="display: inline-block; margin: 0 20px;">3D object detection using estimated depth</strong>
</p>

<div align="center"> 
  <img src="./resources/cre_3d_2.gif" alt="Bottom Right Video" width="600"/> 
</div>



## Performance study

This repository examines how accurate HitNet (Hierarchical Triangle Network) is for 3D object detection, depth estimation, and execution time, and compares it with [CREStereo](https://arxiv.org/abs/2203.11483).

Both methods perform better for depths under 20 meters, which is expected since cameras estimate depth more accurately for closer objects. HitNet shows much higher accuracy for depths less than 5 meters. The HitNet paper mentions it's designed for real-time depth estimation, and the performance study confirms it has a much shorter execution time compared to CREStereo.  

<div align="center">
  <strong style="display: inline-block; margin: 0 20px;">HitNet disparity map estimation and depth calucation</strong> 
</div>

<div align="center"> 
  <img src="./resources/hitnet.jpg" alt="Bottom Left Video" width="600"/> 
</div>

<p align="center">
  <strong style="display: inline-block; margin: 0 20px;">CREStereo disparity map estimation and depth calucation</strong>
</p>

<div align="center"> 
  <img src="./resources/cre.jpg" alt="Bottom Right Video" width="600"/> 
</div>


## Features

- Implementation of HitNet for depth estimation
- Extension of HitNet for 3D object detection
- Evaluation scripts for measuring accuracy
- Comparison with state-of-the-art methods

## Installation

```bash
git clone https://github.com/yourusername/hitnet-3d-detection.git
cd hitnet-3d-detection
pip install -r requirements.txt
```

## Usage

[Provide instructions on how to run your code, including any scripts for training, evaluation, and inference]

## Results

[Summarize your findings on the accuracy of HitNet for 3D object detection and depth estimation. Include any relevant metrics, comparisons, or visualizations]

## Directory Structure

```
hitnet-3d-detection/
├── data/
├── src/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   └── evaluate.py
├── results/
├── requirements.txt
└── README.md
```


## Acknowledgments

(https://github.com/MJITG/PyTorch-HITNet-Hierarchical-Iterative-Tile-Refinement-Network-for-Real-time-Stereo-Matching/tree/main)
https://github.com/megvii-research/CREStereo

## Contact

[Provide your contact information or ways for users to reach out with questions or feedback]
