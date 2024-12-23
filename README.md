# Ship Segmentation on Satellite Images using various U-Net models

## Members

- Đào Quý An - 21020602
- Dương Quang Minh - 21020219
- Lê Văn Bảo - 21020171
- Tống Minh Trí - 21020249

## Description

Mid-Term Project for Computer Vision Class INT3412E 20

This repo was made by UET-VNU students

Topic: Semantic Segmentation on Satellite Imagery Dataset, namely Airbus Ship Detection Dataset (Dataset: [Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection/data)).

Main Technologies: PyTorch, Lightning, Hydra, WanDB

Some implemented models: U-Net, U-Net with ResNet backbone, R2U-Net, Attention U-Net, R2-Attention U-Net, TransU-Net, U-Net++, U-Net3+

Some implemented loss functions: LossBinary (BCE + Jaccard), Dice Loss, Focal Loss, Mixed Loss (Dice + Focal), Lovasz Loss,...

## Obtained results

Because of limitations in time and computing resources, our data configs are:

- Undersample: 140,000
- Subset: 12,000
- Resize: 256x256
- Batch size: 4

Here are our results obtained from experiments:

- Table 1: Performance of different U-Net architectures with Focal Loss

| Model             | Dice       | Jaccard    |
| ----------------- | ---------- | ---------- |
| U-Net             | 0.7096     | 0.6449     |
| U-Net34           | 0.8107     | 0.7467     |
| U-Net50           | 0.7882     | 0.7377     |
| **U-Net101**      | **0.8273** | **0.7596** |
| R2U-Net           | 0.7214     | 0.6770     |
| Attention U-Net   | 0.7622     | 0.7098     |
| Attention R2U-Net | 0.6485     | 0.5990     |
| TransUnet         | 0.6819     | 0.6184     |
| UNet++            | 0.8005     | 0.7362     |
| UNet3+            | 0.7784     | 0.7189     |

- Table 2: Performance of different loss functions with U-Net34 architecture

| Loss               | Dice       | Jaccard    |
| ------------------ | ---------- | ---------- |
| Loss Binary        | 0.8120     | 0.6773     |
| Focal Loss         | 0.8107     | 0.7467     |
| Dice Loss          | 0.811      | 0.6824     |
| Log-Cosh Dice Loss | 0.7194     | 0.6560     |
| Mixed Loss         | 0.8195     | 0.6961     |
| Lovasz Loss        | 0.8675     | 0.7725     |
| **BCE Lovasz**     | **0.8735** | **0.7810** |
| Posweight Lovasz   | 0.8691     | 0.7665     |

- Table 3: Other architecture - loss function combinations

| Model - Loss              | Dice       | Jaccard    |
| ------------------------- | ---------- | ---------- |
| U-Net - Loss Binary       | 0.7644     | 0.6172     |
| U-Net - Focal Loss        | 0.7096     | 0.6449     |
| U-Net - LogCosh Dice Loss | 0.7454     | 0.5944     |
| U-Net - Mixed Loss        | 0.7593     | 0.6127     |
| **U-Net101 - BCE Lovasz** | **0.8852** | **0.7999** |
| TransUnet - Dice Loss     | 0.7590     | 0.6115     |

## How to use this project

- Clone this repo: `git clone https://github.com/daoquyan2003/ship-segmentation.git`

- Make sure Python is installed (Optional: Anaconda `conda create -n airbus` and `conda activate airbus`)

- Run `pip install -r requirements.txt` to install necessary dependencies

- You can also set up on Docker using Dockerfile and docker-compose

- The configuration can be modified using [Hydra](https://hydra.cc/)

- Run `python src/train.py` to start training the model.

## Optional: Deploy with Triton and CVAT

- Checkpoints model U-Net34 + Lovasz Loss + size 768x768 + subset 12000: [Checkpoint](https://drive.google.com/file/d/1BKEtCd3AjOtt2u7DYAUlouIiYQILgCPL/view?usp=sharing)
- Read README.md in folder triton/ and cvat/ for more details
