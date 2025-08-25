# PaR
### Official code implementation of Volumetric Axial Disentanglement Enabling Advancing in Medical Image Segmentation

# Method Details
**We proposed PaR is a plug-and-play module that uses time-axial disentangling to enhance any volumetric segmentation. Its detailed structure is shown in Figure 1.**
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/figures/PaR.png">
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

## Quantitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/figures/fig1.png">
</div>
<p align=center>
  Table 1: Quantitative results on the FALRE2021, OIMHS, and SegTHOR datasets.
</p>

## Qualitative Results
| Methods | Params | FLOPs | FLARE2021 (Dice) | OIMHS (Dice) | SegTHOR (Dice) |
|:--|--:|--:|--:|--:|--:|
| 3D U-Net | 5.75 | 135.88 | 93.08 | 92.49 | 87.59 |
| +PaR | 5.86 | 141.85 | 93.89 | 93.08 | 89.82 |
| V-Net | 45.61 | 333.10 | 89.89 | 88.53 | 85.12 |
| +PaR | 45.72 | 337.26 | 91.49 | 90.26 | 85.81 |
| RAUNet | 70.69 | 366.89 | 93.08 | 91.14 | 88.13 |
| +PaR | 70.80 | 373.06 | 93.28 | 92.25 | 89.08 |
| ResUNet | 27.22 | 902.04 | 92.56 | 90.84 | 88.26 |
| +PaR | 27.33 | 908.01 | 92.94 | 92.79 | 88.71 |
| SegResNet | 4.70 | 61.32 | 91.81 | 90.52 | 86.99 |
| +PaR | 4.81 | 67.29 | 93.13 | 91.35 | 89.60 |
| MultiResUNet | 18.65 | 324.14 | 91.35 | 92.44 | 88.53 |
| +PaR | 18.76 | 330.11 | 91.85 | 93.49 | 89.10 |
| UNETR | 92.62 | 82.58 | 90.70 | 89.05 | 84.03 |
| +PaR | 92.73 | 88.55 | 91.48 | 90.55 | 84.18 |
| Swin UNETR | 61.99 | 329.46 | 93.23 | 92.82 | 87.26 |
| +PaR | 62.10 | 335.63 | 94.04 | 93.27 | 87.42 |
| TransBTS | 30.62 | 110.12 | 92.84 | 87.39 | 86.88 |
| +PaR | 30.74 | 116.29 | 93.27 | 90.55 | 89.13 |
| nnFormer | 149.1 | 224.36 | 91.43 | 88.29 | 86.65 |
| +PaR | 149.21 | 230.53 | 93.69 | 91.80 | 87.69 |
| 3D UX-NET | 53.00 | 627.90 | 93.31 | 93.01 | 87.34 |
| +PaR | 53.11 | 637.93 | 93.84 | 93.66 | 87.77 |


## License
**This project is licensed under the [MIT license](https://github.com/IMOP-lab/PaR-Pytorch/blob/main/LICENSE).**

## Contributors
**The PaR project was implemented with the help of the following contributors:**

Xingru Huang, Jian Huang, Yihao Guo, Tianyun Zhang, Zhao Huang, Yaqi Wang, Ruipu Tang, Guangliang Cheng, Shaowei Jiang, Zhiwen Zheng, Jin Liu, Renjie Ruan, Xiaoshuai Zhang.





