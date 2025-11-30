import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, k=3, p=1):
    """
    Builds a convolutional block with batch normalization and ReLU activation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (int, optional): Kernel size. Defaults to 3.
        p (int, optional): Padding size. Defaults to 1.

    Returns:
        nn.Sequential: Convolutional block with batch normalization and ReLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class segnet_decoder_head(nn.Module):
    """
    SegNet decoder that mirrors VGG16-BN encoder using MaxUnpool2d.
    Expects forward input as (y5, indices_list, size_list) from vgg16_encoder.
    Produces raw logits with num_classes channels.
    """

    def __init__(self, num_classes):
        """
        Initializes SegNet decoder with MaxUnpool2d and convolutional blocks.

        Args:
            num_classes (int): Number of output classes.

        """
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.dec5_convs = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.dec4_convs = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 256),
        )
        self.dec3_convs = nn.Sequential(
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 128),
        )
        self.dec2_convs = nn.Sequential(
            _conv_bn_relu(128, 128),
            _conv_bn_relu(128, 64),
        )
        self.dec1_convs = nn.Sequential(
            _conv_bn_relu(64, 64),
            _conv_bn_relu(64, 64),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _up_block(self, x, idx, out_size, convs):
        """
        Helper function to upsample and convolve a feature map.

        Args:
            x (torch.Tensor): Feature map to be upscaled.
            idx (torch.Tensor): Indices of the maxpool operation.
            out_size (tuple): Desired output size.
            convs (nn.Sequential): Convolutional block to be applied after upscaling.

        Returns:
            torch.Tensor: Upscaled and convolved feature map.
        """
        x = self.unpool(x, indices=idx, output_size=out_size)
        x = convs(x)
        return x

    def forward(self, enc_tuple):
        """
        Forward pass of the SegNet decoder.

        Args:
            enc_tuple (tuple): Encoder output, containing feature map, indices of maxpool, and sizes of the feature maps at each encoder stage.

        Returns:
            torch.Tensor: Logits of the segmentation mask.
        """
        y5, indices, sizes = enc_tuple
        i5, i4, i3, i2, i1 = indices
        s5, s4, s3, s2, s1 = sizes

        d5 = self._up_block(y5, i5, s5, self.dec5_convs)
        d4 = self._up_block(d5, i4, s4, self.dec4_convs)
        d3 = self._up_block(d4, i3, s3, self.dec3_convs)
        d2 = self._up_block(d3, i2, s2, self.dec2_convs)
        d1 = self._up_block(d2, i1, s1, self.dec1_convs)

        logits = self.classifier(d1)
        return logits
