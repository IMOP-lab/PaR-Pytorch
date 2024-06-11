# PaR
### Official code implementation of Volumetric Axial Disentanglement

### [Project page](https://github.com/IMOP-lab/PaR-Pytorch) | [Our laboratory home page](https://github.com/IMOP-lab)

# Method Details
**We proposed PaR is a plug-and-play module that uses time-axial disentangling to enhance any volumetric segmentation. Its detailed structure is shown in Figure 1.**
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/figures/PaR.png"width=80% height=80%>
</div>
<p align=center>
  Figure 1: Detailed module structure of the PaR.
</p>

**As a plug-and-play module, we'll show you how to easily integrate PaR into any network architecture.**
 
# To Start
## Environment
    python = 3.9.0
    pytorch = 2.0.0+cu118
    monai = 0.9.0
    numpy = 1.23.2

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
  Figure 1: Quantitative results on the FALRE2021, OIMHS, and SegTHOR datasets.
</p>

## Qualitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/PaR-Pytorch/blob/main/tables/PaR_cmp_result.png">
</div>
<p align=center>
  Figure 2: Qualitative results on the FALRE2021, OIMHS, and SegTHOR datasets.
</p>

## Ablation Study
