import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


def _conv_bn_relu(in_ch, out_ch, k=3, p=1):
    """
    Builds a Conv -> BN -> ReLU block.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (int): Kernel size of the convolutional layer. Defaults to 3.
        p (int): Padding of the convolutional layer. Defaults to 1.

    Returns:
        nn.Sequential: A Conv -> BN -> ReLU block.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class vgg16_encoder(nn.Module):
    """
    VGG16-BN style encoder for SegNet.
    Uses MaxPool2d(return_indices=True) to pass pooling indices to the decoder.
    Returns a tuple: (y5, indices_list, size_list), where lists are ordered [i5, i4, i3, i2, i1] and [s5, s4, s3, s2, s1].
    """

    def __init__(self, in_chans=3, pretrained=True, freeze_bn=False):
        """
        Initializes the VGG16-BN encoder for SegNet.

        Args:
            in_chans (int): Number of input channels.
            pretrained (bool): If True, initializes the weights from a pre-trained VGG16-BN model.
            freeze_bn (bool): If True, freezes the BatchNorm2d layers.

        """
        super().__init__()

        # VGG16-BN channel progression
        # Enc1: 2 convs -> 64
        self.enc1_convs = nn.Sequential(
            _conv_bn_relu(in_chans, 64),
            _conv_bn_relu(64, 64),
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        # Enc2: 2 convs -> 128
        self.enc2_convs = nn.Sequential(
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Enc3: 3 convs -> 256
        self.enc3_convs = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Enc4: 3 convs -> 512
        self.enc4_convs = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Enc5: 3 convs -> 512
        self.enc5_convs = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        if pretrained:
            self._init_from_torchvision(in_chans=in_chans)

        if freeze_bn:
            self.freeze_bn()

    @torch.no_grad()
    def _init_from_torchvision(self, in_chans=3):
        """
        Initializes the weights of the VGG16-BN encoder from a pre-trained VGG16-BN model.

        Args:
            in_chans (int): Number of input channels.

        Modifies the weights of the VGG16-BN encoder to match the pre-trained VGG16-BN model.
        """
        tv = vgg16_bn(weights="IMAGENET1K_V1")
        feats = tv.features

        # Helper to copy Conv+BN weights
        def copy_block(src_seq, dst_seq, conv_bn_count):
            """
            Copies weights from a source sequence to a destination sequence.

            Args:
                src_seq (nn.Sequential): Source sequence to copy weights from.
                dst_seq (nn.Sequential): Destination sequence to copy weights to.
                conv_bn_count (int): Number of Conv+BatchNorm groups to copy.

            Copies weights from the source sequence to the destination sequence. Handles in_channels mismatch for the first conv only.

            """
            si = 0
            for i in range(conv_bn_count):
                conv_ours = dst_seq[3 * i + 0]
                bn_ours = dst_seq[3 * i + 1]
                while si < len(src_seq) and not isinstance(src_seq[si], nn.Conv2d):
                    si += 1
                conv_tv = src_seq[si]
                si += 1
                while si < len(src_seq) and not isinstance(src_seq[si], nn.BatchNorm2d):
                    si += 1
                bn_tv = src_seq[si]
                si += 1
                if si < len(src_seq) and isinstance(src_seq[si], nn.ReLU):
                    si += 1

                # Handle in_chans mismatch for the first conv only
                if conv_ours.in_channels != conv_tv.in_channels:
                    w = conv_tv.weight
                    if conv_ours.in_channels == 1:
                        w_new = w.mean(dim=1, keepdim=True)
                    else:
                        rep = (conv_ours.in_channels + w.shape[1] - 1) // w.shape[1]
                        w_new = w.repeat(1, rep, 1, 1)[:, : conv_ours.in_channels, :, :]
                    conv_ours.weight.copy_(w_new)
                else:
                    conv_ours.weight.copy_(conv_tv.weight)

                bn_ours.weight.copy_(bn_tv.weight)
                bn_ours.bias.copy_(bn_tv.bias)
                bn_ours.running_mean.copy_(bn_tv.running_mean)
                bn_ours.running_var.copy_(bn_tv.running_var)

        ours = [
            (self.enc1_convs, 2),
            (self.enc2_convs, 2),
            (self.enc3_convs, 3),
            (self.enc4_convs, 3),
            (self.enc5_convs, 3),
        ]

        si = 0
        for enc_block, nconvs in ours:
            start = si
            while si < len(feats) and not isinstance(feats[si], nn.MaxPool2d):
                si += 1
            slice_seq = feats[start:si]
            copy_block(slice_seq, enc_block, nconvs)
            if si < len(feats) and isinstance(feats[si], nn.MaxPool2d):
                si += 1

        del tv

    def freeze_bn(self):
        """
        Freeze all BatchNorm2d layers in the model, i.e. set them to eval mode and require their parameters to be non-trainable.
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the SegNet encoder.

        Args:
            x (torch.Tensor): Input image, shape (N, C, H, W).

        Returns:
            tuple: A tuple containing the output of the last encoder stage, indices of maxpool at each stage, and sizes of the feature maps at each encoder stage.
        """
        x1 = self.enc1_convs(x)
        y1, i1 = self.pool1(x1)
        x2 = self.enc2_convs(y1)
        y2, i2 = self.pool2(x2)
        x3 = self.enc3_convs(y2)
        y3, i3 = self.pool3(x3)
        x4 = self.enc4_convs(y3)
        y4, i4 = self.pool4(x4)
        x5 = self.enc5_convs(y4)
        y5, i5 = self.pool5(x5)

        indices = [i5, i4, i3, i2, i1]
        sizes = [x5.size(), x4.size(), x3.size(), x2.size(), x1.size()]
        return y5, indices, sizes
