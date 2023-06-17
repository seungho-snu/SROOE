[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fseungho-snu%2FSROOE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/perception-oriented-single-image-super/image-super-resolution-on-div2k-val-4x)](https://paperswithcode.com/sota/image-super-resolution-on-div2k-val-4x?p=perception-oriented-single-image-super)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/perception-oriented-single-image-super/image-super-resolution-on-general100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-general100-4x?p=perception-oriented-single-image-super)

# SROOE

# Perception-Oriented Single Image Super-Resolution using Optimal Objective Estimation (CVPR 2023)
paper <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Park_Perception-Oriented_Single_Image_Super-Resolution_Using_Optimal_Objective_Estimation_CVPR_2023_paper.html">Link</a>

Seung Ho Park, Young Su Moon, Nam Ik Cho

## Abstract
Single-image super-resolution (SISR) networks trained with perceptual and adversarial losses provide high-contrast outputs compared to those of networks trained with distortion-oriented losses, such as L1 or L2. However, it has been shown that using a single perceptual loss is insufficient for accurately restoring locally varying diverse shapes in images, often generating undesirable artifacts or unnatural details. For this reason, combinations of various losses, such as perceptual, adversarial, and distortion losses, have been attempted, yet it remains challenging to find optimal combinations. Hence, in this paper, we propose a new SISR framework that applies optimal objectives for each region to generate plausible results in overall areas of high-resolution outputs. Specifically, the framework comprises two models: a predictive model that infers an optimal objective map for a given low-resolution (LR) input and a generative model that applies a target objective map to produce the corresponding SR output. The generative model is trained over our proposed objective trajectory representing a set of essential objectives, which enables the single network to learn various SR results corresponding to combined losses on the trajectory. The predictive model is trained using pairs of LR images and corresponding optimal objective maps searched from the objective trajectory. Experimental results on five benchmarks show that the proposed method outperforms state-of-the-art perception-driven SR methods in LPIPS, DISTS, PSNR, and SSIM metrics. The visual results also demonstrate the superiority of our method in perception-oriented reconstruction.
<br><br>

## Usage:

### Environments
- Pytorch 1.10.0
- CUDA 11.3
- Python 3.8

### Visual and quantitative comparison. 
<p align="center"><img src="figures/Fig1.PNG" width="500"></p>
The proposed SROOE shows a higher PSNR, LRPSNR and lower LPIPS than other state-of-the-art methods, i.e, lower distortion and higher perceptual quality.

### Quick usage on your data:

    python test.py -opt options/test/test_SROOE_4x.yml
    
### Test models

- Download the pretrained SR 4x model <a href="https://www.dropbox.com/s/v7lx9qoji1ndonx/SR.pth?dl=0">Link</a>   
- Download the pretrained OOE model <a href="https://www.dropbox.com/s/hoykbrpadzozlab/OOE.pth?dl=0">Link</a>   

### Training
- Training codes for SROT model <a href="https://github.com/seungho-snu/SROT/blob/main/README.md">Link</a>

## Experimental Results

### Quantitative Evaluation

<p align="center"><img src="figures/table2.PNG" width="800"></p>

### Visual Evaluation

Visual comparison with state-of-the-art perception-driven SR methods

<p align="center"><img src="figures/figure-01.PNG" width="800"></p>

<p align="center"><img src="figures/figure-02.PNG" width="800"></p>

<p align="center"><img src="figures/figure-03.PNG" width="800"></p>

<p align="center"><img src="figures/figure-04.PNG" width="800"></p>

<p align="center"><img src="figures/figure-05.PNG" width="800"></p>

<p align="center"><img src="figures/figure-06.PNG" width="800"></p>

<p align="center"><img src="figures/figure-07.PNG" width="800"></p>

# Citation

    @misc{https://doi.org/10.48550/arxiv.2211.13676,
      doi = {10.48550/ARXIV.2211.13676},
      url = {https://arxiv.org/abs/2211.13676},
      author = {Park, Seung Ho and Moon, Young Su and Cho, Nam Ik},
      title = {Perception-Oriented Single Image Super-Resolution using Optimal Objective Estimation},
      publisher = {arXiv},
      year = {2022},  
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
    
    @InProceedings{Park_2023_CVPR,
        author    = {Park, Seung Ho and Moon, Young Su and Cho, Nam Ik},
        title     = {Perception-Oriented Single Image Super-Resolution Using Optimal Objective Estimation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {1725-1735}
    }



# Acknowledgement
Our work and implementations are inspired by and based on BasicSR <a href="https://github.com/xinntao/BasicSR">[site]</a> 
