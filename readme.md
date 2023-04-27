The code sources of our evaluation model method in the LIDC-IDRI and Luna16 datasets are as follows:

| Model  name         | Access  link                                                                 |
| ------------------- | ---------------------------------------------------------------------------- |
| 2D  segmented model |                                                                              |
| UNet                | -                                                                            |
| UNet++              | https://github.com/qubvel/segmentation_models.pytorch                        |
| UNet3+              | https://github.com/UCAS-Vincent/UNet-3Plus                                   |
| Cpfnet              | https://github.com/FENGShuanglang/CPFNet_Project                             |
| Raunet              | https://github.com/nizhenliang/RAUNet                                        |
| BioNet              | https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo                |
| Swin unet           | https://github.com/microsoft/Swin-Transformer                                |
| Unext               | https://github.com/jeya-maria-jose/UNeXt-pytorch                             |
| SGUNet              | -                                                                            |
| UCTransNet          | https://github.com/McGregorWwww/UCTransNet                                   |
| UTNet               | https://github.com/yhygao/UTNet                                              |
| 3D  segmented model |                                                                              |
| Unet                | -                                                                            |
| ResUnet             | -                                                                            |
| Vnet                | https://github.com/MIILab-MTU/AIInHealthcare_LVSeg                           |
| Unet++              | https://github.com/lose4578/nnUNet_plusplus                                  |
| Ynet                | -                                                                            |
| ReconNet            | https://github.com/Chinmayrane16/ReconNet-PyTorch                            |
| Unetr               | https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV |
| TransBTS            | https://github.com/Wenxuan-1119/TransBTS                                     |
| VTUNet              | https://github.com/himashi92/VT-UNet                                         |
| WingsNet            | https://github.com/haozheng-sjtu/3d-airway-segmentation                      |
| PCAMNET             | https://github.com/PerceptionComputingLab/PCAMNet/                           |
| ASA                 | https://github.com/lhaof/asa                                                 |

Note: " -" means that due to other factors, the corresponding github warehouse cannot be found for the time being. But you can access the model source code used in this experiment through the following link(https://github.com/BITEWKRER/exp-models). At the same time, we also open source 2D and 3D pre -processing data through the following links().

 The evaluation metric for this experiment was used from [(TorchMetrics)](https://torchmetrics.readthedocs.io/en/stable/)
