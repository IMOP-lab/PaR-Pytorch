# PaR
Official code implementation of Volumetric Axial Disentanglement

## To Start
Take 3D U-Net for example:

    from model.3dunet import UNetModule
    from model.PaR import PaR
    
    class 3DUNet_PaR(nn.Module):
        def __init__(self, out_classes):
            super().__init__()
            self.out_classes = out_classes
            # Initialize the 3D-UNet
            self.3DUNet = UNetModule(in_channels=1,
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
            x1 = self.3DUNet(x)
            x2 = self.par(x1)
            x3 = self.sigmoid(x2)
            out = x3 * x1 + x1
            return out
