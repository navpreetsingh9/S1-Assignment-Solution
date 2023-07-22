# S10-Assignment-Solution

This repository contains a custom implementation of the ResNet architecture for the CIFAR10 dataset. The ResNet architecture is  designed to achieve a target accuracy of 90% on the CIFAR10 dataset  using the One Cycle Policy for training.



## Files Description

The repository consists of the following files:

### 1. ERA-S10.ipynb

This Jupyter Notebook file contains the main code and instructions to run the training process for the custom ResNet model on the CIFAR10 dataset. It imports the necessary modules from the other files and runs the dataloading, torchsummary, lr_finder to get min and max lr and finally trains the model for 24 epochs and shows the validation accuracy.

### 2. dataloader.py

The `dataloader.py` file contains the code to create the data loader for the CIFAR10 dataset. It handles loading the dataset, and creating batches of data for training and validation.

### 3. model.py

The `model.py` file defines the CustomResNet architecture for CIFAR10. It includes the `PrepLayer`, `Layer1`, `Layer2`, `Layer3`, and the fully connected (`FC`) layer, with appropriate activation functions and batch normalization.

### 4. train.py

The `train.py` file contains the training loop for the custom ResNet model. It utilizes the One Cycle Policy and trains the model for the specified number of epochs with the ADAM optimizer and CrossEntropyLoss.

### 5. utils.py

The `utils.py` file contains utility functions used in the training process, such as functions to get the lr, visualize images, get device, print model summary, and other helper functions.

### 6. transforms.py

The `transforms.py` file includes data augmentation and transformation functions used during training. It contains the `RandomCrop`, `FlipLR`, and `CutOut` transformations as specified in the assignment.

## Architecture

The Custom ResNet architecture for CIFAR10 is structured as follows:

==========================================================================================

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomResNet                             [1, 10]                   --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 64, 32, 32]           1,728
│    └─ReLU: 2-2                         [1, 64, 32, 32]           --
│    └─BatchNorm2d: 2-3                  [1, 64, 32, 32]           128
│    └─Dropout: 2-4                      [1, 64, 32, 32]           --
├─ResNetBlock: 1-2                       [1, 128, 16, 16]          --
│    └─Sequential: 2-5                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-1                  [1, 128, 32, 32]          73,728
│    │    └─MaxPool2d: 3-2               [1, 128, 16, 16]          --
│    │    └─ReLU: 3-3                    [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-4             [1, 128, 16, 16]          256
│    │    └─Dropout: 3-5                 [1, 128, 16, 16]          --
│    └─Sequential: 2-6                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-6                  [1, 128, 16, 16]          147,456
│    │    └─ReLU: 3-7                    [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-8             [1, 128, 16, 16]          256
│    │    └─Dropout: 3-9                 [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-10                 [1, 128, 16, 16]          147,456
│    │    └─ReLU: 3-11                   [1, 128, 16, 16]          --
│    │    └─BatchNorm2d: 3-12            [1, 128, 16, 16]          256
├─Sequential: 1-3                        [1, 256, 8, 8]            --
│    └─Conv2d: 2-7                       [1, 256, 16, 16]          294,912
│    └─MaxPool2d: 2-8                    [1, 256, 8, 8]            --
│    └─ReLU: 2-9                         [1, 256, 8, 8]            --
│    └─BatchNorm2d: 2-10                 [1, 256, 8, 8]            512
│    └─Dropout: 2-11                     [1, 256, 8, 8]            --
├─ResNetBlock: 1-4                       [1, 512, 4, 4]            --
│    └─Sequential: 2-12                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-13                 [1, 512, 8, 8]            1,179,648
│    │    └─MaxPool2d: 3-14              [1, 512, 4, 4]            --
│    │    └─ReLU: 3-15                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-16            [1, 512, 4, 4]            1,024
│    │    └─Dropout: 3-17                [1, 512, 4, 4]            --
│    └─Sequential: 2-13                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-18                 [1, 512, 4, 4]            2,359,296
│    │    └─ReLU: 3-19                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-20            [1, 512, 4, 4]            1,024
│    │    └─Dropout: 3-21                [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-22                 [1, 512, 4, 4]            2,359,296
│    │    └─ReLU: 3-23                   [1, 512, 4, 4]            --
│    │    └─BatchNorm2d: 3-24            [1, 512, 4, 4]            1,024
├─MaxPool2d: 1-5                         [1, 512, 1, 1]            --
├─Linear: 1-6                            [1, 10]                   5,130
==========================================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
Total mult-adds (M): 379.27
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
==========================================================================================
```



## One Cycle Policy

The training process uses the One Cycle Policy with the following configurations:

- Total Epochs: 24
- Max Learning Rate (LR) at Epoch: 5
- LRMIN: To be determined during training
- LRMAX: To be determined during training
- No Annihilation during the learning rate schedule

## Data Transformations

The dataset is transformed using the following steps:

- RandomCrop of size 32x32 (after padding of 4)
- FlipLR (Flip Left-Right)
- CutOut with a mask size of 8x8

## Training Configuration

- Batch size: 512
- Optimizer: Adam
- Loss function: CrossEntropyLoss

## Code Modularity

The code for the CustomResNet model is modularized and divided into separate files for better organization and reusability. The `ERA-S10.ipynb` notebook serves as the main script to run the training process by importing functions and modules from the other files. The model, data loader, and other utility functions can be found in their respective files.

**Note:** While running the code in the notebook, it will automatically clone the required repository for running the training.