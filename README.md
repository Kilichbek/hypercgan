# HyperCGAN: Adversarial Text to Continuous Image Generation (CVPR 2024)

Official PyTorch implementation of **Adversarial Text to Continuous Image Generation**.

Kilichbek Haydarov, Aashiq Muhamed, Xiaoqian Shen, Jovana Lazarevic, Ivan Skorokhodov, Chamuditha Jayanga Galappaththige, Mohamed Elhoseiny

[[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Haydarov_Adversarial_Text_to_Continuous_Image_Generation_CVPR_2024_paper.html) [[Project page]](https://kilichbek.github.io/webpage/hypercgan)

HyperCGAN represents images as implicit neural representations (INRs) and conditions an INR-based GAN on text through hypernetworks. A word-level attention-based weight modulation operator lets every word of the caption modulate the weights of the INR decoder. The resulting text-to-image model keeps the properties of continuous generative models: it can synthesize pixels beyond the training field of view (extrapolation) and render images at higher resolutions than it was trained on (super-resolution).

## Overview

- **Generator** (`training/networks.py`): an INR-GAN backbone, i.e. a pixel-wise MLP (all layers are 1x1) on Fourier coordinate embeddings (`training/layers.py`). The weight of every layer is modulated
  1. by a rank-10 factorized multiplicative modulation (FMM) predicted from the latent code, and
  2. by the **word-level attention modulation** (`training/tensor_mod.py`): a linear hypernetwork maps every CLIP token embedding to a rank-1 modulation of the layer's weight, and the weight attends over the words to combine them.
- **Text encoder** (`training/text_encoder.py`): frozen CLIP ViT-B/32. The word embeddings are the per-token outputs of its final layer norm; the sentence embedding is the regular CLIP text embedding.
- **Discriminator and losses** (`training/loss.py`): StyleGAN2 discriminator with the image-text contrastive losses of [LAFITE](https://github.com/drboog/Lafite).

| File | Purpose |
| --- | --- |
| `dataset_tool.py` | Converts image-caption pairs into a dataset archive |
| `train.py` | Training (settings of the paper in `scripts/`) |
| `calc_metrics.py` | FID / KID / precision-recall / IS |
| `clip_rprecision.py` | CLIP R-precision |
| `generate.py` | Text-to-image generation, extrapolation and super-resolution |
| `configs/hypercgan.yaml` | Generator architecture |

## Installation

```bash
conda env create -f environment.yml
conda activate hypercgan
```

The paper experiments used Python 3.10 and CUDA 11.7. The custom CUDA kernels of StyleGAN2-ADA are compiled on first use, which requires the CUDA toolkit (`nvcc`) and `ninja`. If compilation fails, a slower reference implementation is used.

## Data preparation

Put every image next to a `.txt` file with the same name that contains its captions, one per line, and convert the folder (or a zip with the same structure) into a dataset archive:

```bash
python dataset_tool.py --source=path/to/cub/train --dest=datasets/cub_train.zip \
    --transform=center-crop --width=256 --height=256
python dataset_tool.py --source=path/to/cub/test --dest=datasets/cub_test.zip \
    --transform=center-crop --width=256 --height=256
```

The archive stores the images as uncompressed PNGs and the captions as CLIP tokens in `dataset.json`. Sub-directories of the source are allowed; `clip_rprecision.py` treats the parent directory of an image as its class, so keep the class folders of CUB (e.g. `001.Black_footed_Albatross/`) to draw negative captions from other classes.

We used the following datasets at 256x256:

| Dataset | Captions | Split | #Images (train / test) |
| --- | --- | --- | --- |
| [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | 10 per image (Reed et al.) | as in [AttnGAN](https://github.com/taoxugit/AttnGAN) | 8,855 / 2,933 |
| [MS-COCO 2014](https://cocodataset.org) | 5 per image | train2014 / val2014 | 82,783 / 40,470 |
| [ArtEmis](https://www.artemisdataset.org) | ArtEmis utterances (WikiArt images) | train / test | 69,213 / 8,160 |

## Training

```bash
bash scripts/train_cub.sh       # CUB: 8 GPUs, ADA augmentation
bash scripts/train_coco.sh      # MS-COCO
bash scripts/train_artemis.sh   # ArtEmis
```

Each script runs `train.py` with the settings of the paper; extra options are forwarded (e.g. `--resume=path/to/network-snapshot.pkl` to continue a run, or `GPUS=4 bash scripts/train_cub.sh --batch=64` for fewer GPUs). Snapshots, image grids (`reals.png` and generated images for the same captions), logs and a training-time FID (computed on the training set) are written to `training-runs/`. Our models were trained for roughly 17-18M images.

Full list of hyper-parameters used in the paper:

| | |
| --- | --- |
| Generator | INR-GAN with 32x32 input grid, FMM rank 10 (`demod`), word modulation with `lrelu`, fp32 |
| Mapping network | 8 layers |
| Batch size | 128 (8 GPUs x 16) |
| Optimizer | Adam, lr 0.0025, betas (0, 0.99); G EMA half-life 20k images |
| GAN regularization | R1 (gamma 10), path length regularization (weight 2), style mixing (prob. 0.9) |
| Contrastive losses | temperature 0.5, CLIP image-text weight `itc`=10, discriminator image-text weight `itd`=5 |
| Augmentation | x-flips; ADA (`bg` pipeline, target 0.6) on CUB only |

## Evaluation

```bash
# FID: 30k images generated from random test captions vs. all real test images.
python calc_metrics.py --network=path/to/network-snapshot.pkl --data=datasets/cub_test.zip --metrics=fid50k_full --gpus=1

# CLIP R-precision (R@1): ground-truth caption vs. 99 captions of other classes, 30k samples, 10 splits.
python clip_rprecision.py --network=path/to/network-snapshot.pkl --data=datasets/cub_test.zip
```

`fid50k_full` keeps its StyleGAN2-ADA name but uses 30k generated images, following the standard text-to-image protocol (see `metrics/metric_utils.py`).

## Text-to-image and continuous generation

```bash
# Text-to-image.
python generate.py --network=path/to/network-snapshot.pkl --outdir=out --seeds=0-7 \
    --prompt="this bird has a red crown, a white belly and a short black beak"

# Extrapolation: render a 1.5x larger field of view around the image (384x384 for a 256x256 model).
python generate.py --network=path/to/network-snapshot.pkl --outdir=out --extrapolate=1.5 \
    --prompt="a living room with a couch and a television"

# Super-resolution: render 1024x1024 images with a 256x256 model.
python generate.py --network=path/to/network-snapshot.pkl --outdir=out --superres=4 \
    --prompt="a painting of a calm lake at sunset"

# One image for each of 30k random test captions, e.g. for external evaluation tools.
python generate.py --network=path/to/network-snapshot.pkl --outdir=out --data=datasets/cub_test.zip --num=30000
```

Extrapolation and super-resolution can be combined. By default, super-resolution starts at the last block that runs at the 32x32 input resolution; `--superres-block-res=4` instead evaluates the whole INR on the denser grid. For example, `--extrapolate=2 --superres=2 --superres-block-res=4` renders a 2x larger field of view at 2x pixel density (1024x1024), which is how our original extrapolation script rendered images.

## Pretrained models

We do not provide pretrained models.

## Citation

```bibtex
@InProceedings{Haydarov_2024_CVPR,
    author    = {Haydarov, Kilichbek and Muhamed, Aashiq and Shen, Xiaoqian and Lazarevic, Jovana and Skorokhodov, Ivan and Galappaththige, Chamuditha Jayanga and Elhoseiny, Mohamed},
    title     = {Adversarial Text to Continuous Image Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6316-6326}
}
```

## Acknowledgements

This code builds on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [LAFITE](https://github.com/drboog/Lafite), [INR-GAN](https://github.com/universome/inr-gan) and [CLIP](https://github.com/openai/CLIP).

## License

This repository is a derivative of StyleGAN2-ADA and is therefore distributed under the [NVIDIA Source Code License](LICENSE.txt), which permits non-commercial (research or evaluation) use only. Parts derived from LAFITE are additionally covered by the [MIT license](licenses/LICENSE-LAFITE.txt).
