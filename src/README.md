## `src` directory Structure

The `src` directory structure of our project looks like this:

```
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   │   ├── airbus                     <- Folder containing airbus (this project) data scripts
│   │   │   ├── components                   <- Dataset scripts for airbus data
│   │   │   │   ├── airbus.py                      <- Dataset class for preparing data, loading images and masks, etc.
│   │   │   │   ├── transform_airbus.py            <- Dataset class for transforming data, i.e. add augmentations to images and masks
│   │   │   ├── airbus_datamodule.py         <- Lightning Datamodule class for splitting data into train/val/test sets, and preparing
│   │   │                                       dataloaders
│   ├── models                   <- Model scripts
│   │   ├── loss_function              <- Folder containing different loss functions for experiments
│   │   ├── unet                       <- Model scripts for U-Net and its variants
│   │   │   ├── components                   <- Folder containing different U-Net architectures, i.e. U-Net, U-Net++, U-Net3+, etc.
│   │   │   ├── unet_module.py               <- Lightning Module for training, validating and testing logics
│   ├── utils                    <- Utility scripts
│   │   ├── callback                   <- Callback scripts (i.e. for logging results - images and predicted masks - on logger)
│   │
│   ├── eval.py                  <- Script to run evaluation
│   └── train.py                 <- Script to run training

```
