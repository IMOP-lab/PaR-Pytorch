# PaR
### Official code implementation of Volumetric Axial Disentanglement Enabling Advancing in Medical Image Segmentation

# Method Details
**We proposed PaR is a plug-and-play module that uses time-axial disentangling to enhance any volumetric segmentation. Its detailed structure is shown in Figure 1.**
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/assets/PaR.png">
</div>
<p align=center>
  Fig. 1: Detailed module structure of the PaR.
</p>

**As a plug-and-play module, we'll show you how to easily integrate PaR into any network architecture.**
 
# To Start
## Environment
    python = 3.9.0
    pytorch = 2.0.0+cu118
    monai = 0.9.0
    numpy = 1.23.2
**For a full list of software packages and version numbers, please take a look at the experimental environment file [environment.yaml](https://github.com/IMOP-lab/PaR-Pytorch/blob/main/environment.yaml).**

## Take 3D U-Net for example:
    
    class UNet3D_PaR(nn.Module):
        def __init__(self, out_classes):
            super().__init__()
            self.out_classes = out_classes
            # Initialize the 3D-UNet
            self.UNet3D = UNetModule(in_channels=1,
                out_channels=self.out_classes,
                lr_rate=1e-4,
                class_weight=None,
                sw_batch_size=1,
                cls_loss="DiceCE",
                val_patch_size=(96,96,96),
                overlap=0.75,
                val_frequency=100,
                weight_decay=2e-6)

        self.par = PaR(axial_dim=96, in_channels=self.out_classes, heads=8, groups=8)
        self.sigmoid= nn.Sigmoid()

        def forward(self, x):
            x1 = self.UNet3D(x)
            x2 = self.par(x1)
            x3 = self.sigmoid(x2)
            out = x3 * x1 + x1
            return out

## Take UNETR for example:
    
    class UNETR_PaR(nn.Module):
        def __init__(self, out_classes):
            super().__init__()
            self.out_classes = out_classes
            # Initialize the UNETR
            self.UNETR = UNETR(
                in_channels=1,
                out_channels=self.out_classes,
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0)

        self.par = PaR(axial_dim=96, in_channels=self.out_classes, heads=8, groups=8)
        self.sigmoid= nn.Sigmoid()

        def forward(self, x):
            x1 = self.UNETR(x)
            x2 = self.par(x1)
            x3 = self.sigmoid(x2)
            out = x3 * x1 + x1
            return out

# Experiment 
**Here, we present the quantitative and qualitative results of different benchmark models on the public data sets FALRE2021, OIMHS, SegTHOR, and the improvement of the segmentation effect brought by the introduction of PaR.**


## Qualitative Results
| Methods | Params | FLOPs | FLARE2021 (Dice) | FLARE2021 (HD95) | OIMHS (Dice) | OIMHS (HD95) | SegTHOR (Dice) | SegTHOR (HD95) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 3D U-Net | 5.75 | 135.88 | 93.08 | 16.31 | 92.49 | 3.40 | 87.59 | 4.32 |
| +PaR | 5.86 | 141.85 | 93.89 | 2.49 | 93.08 | 2.89 | 89.82 | 2.98 |
| V-Net | 45.61 | 333.10 | 89.89 | 12.93 | 88.53 | 18.13 | 85.12 | 16.10 |
| +PaR | 45.72 | 337.26 | 91.49 | 5.98 | 90.26 | 13.52 | 85.81 | 11.49 |
| RAUNet | 70.69 | 366.89 | 93.08 | 27.37 | 91.14 | 13.61 | 88.13 | 14.58 |
| +PaR | 70.80 | 373.06 | 93.28 | 26.71 | 92.25 | 5.37 | 89.08 | 2.99 |
| ResUNet (2019) | 27.22 | 902.04 | 92.56 | 30.15 | 90.84 | 3.92 | 88.26 | 3.22 |
| +PaR | 27.33 | 908.01 | 92.94 | 11.80 | 92.79 | 3.33 | 88.71 | 3.14 |
| SegResNet | 4.70 | 61.32 | 91.81 | 3.22 | 90.52 | 12.05 | 86.99 | 3.38 |
| +PaR | 4.81 | 67.29 | 93.13 | 2.78 | 91.35 | 5.07 | 89.60 | 2.87 |
| MultiResUNet | 18.65 | 324.14 | 91.35 | 9.04 | 92.44 | 3.23 | 88.53 | 26.75 |
| +PaR | 18.76 | 330.11 | 91.85 | 3.68 | 93.49 | 2.85 | 89.10 | 11.06 |
| UNETR (2021) | 92.62 | 82.58 | 90.70 | 4.63 | 89.05 | 29.15 | 84.03 | 4.71 |
| +PaR | 92.73 | 88.55 | 91.48 | 3.64 | 90.55 | 7.24 | 84.18 | 4.65 |
| Swin UNETR (2021) | 61.99 | 329.46 | 93.23 | 3.25 | 92.82 | 5.21 | 87.26 | 3.87 |
| +PaR | 62.10 | 335.63 | 94.04 | 2.61 | 93.27 | 2.92 | 87.42 | 3.62 |
| TransBTS | 30.62 | 110.12 | 92.84 | 3.54 | 87.39 | 33.52 | 86.88 | 3.84 |
| +PaR | 30.74 | 116.29 | 93.27 | 2.90 | 90.55 | 21.00 | 89.13 | 3.75 |
| nnFormer | 149.1 | 224.36 | 91.43 | 5.41 | 88.29 | 25.32 | 86.65 | 5.11 |
| +PaR | 149.21 | 230.53 | 93.69 | 2.35 | 91.80 | 7.36 | 87.69 | 3.51 |
| 3D UX-NET | 53.00 | 627.90 | 93.31 | 8.85 | 93.01 | 4.61 | 87.34 | 4.69 |
| +PaR | 53.11 | 637.93 | 93.84 | 2.43 | 93.66 | 2.65 | 87.77 | 3.61 |


## Quantitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/assets/qualitative_results.png">
</div>
<p align=center>
  Table 1: Quantitative results on the FALRE2021, OIMHS, and SegTHOR datasets.
</p>

## License
**This project is licensed under the [MIT license](https://github.com/IMOP-lab/PaR-Pytorch/blob/main/LICENSE).**

## Contributors
**The PaR project was implemented with the help of the following contributors:**

Xingru Huang, Jian Huang, Yihao Guo, Tianyun Zhang, Zhao Huang, Yaqi Wang, Ruipu Tang, Guangliang Cheng, Shaowei Jiang, Zhiwen Zheng, Jin Liu, Renjie Ruan, Xiaoshuai Zhang.





