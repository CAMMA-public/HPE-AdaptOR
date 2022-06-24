# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
import torch.nn as nn
from detectron2.config import configurable
from detectron2.modeling.roi_heads import ROI_KEYPOINT_HEAD_REGISTRY
from detectron2.modeling.roi_heads.keypoint_head import BaseKeypointRCNNHead
from detectron2.layers import Conv2d, ConvTranspose2d, interpolate, get_norm
import fvcore.nn.weight_init as weight_init


__all__ = ["KRCNNConvDeconvUpsampleHeadGN"]

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHeadGN(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, conv_norm, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
        super().__init__(num_keypoints=num_keypoints, **kwargs)

        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0
        in_channels = input_shape.channels
        
        self.conv_norm_relus = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(
                in_channels,
                layer_channels,
                3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, layer_channels),
            )
            self.add_module("conv_fcn{}".format(idx), module)
            self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                if "norm" not in name:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                    

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        ret["conv_norm"] = cfg.MODEL.ROI_KEYPOINT_HEAD.NORM
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        x = interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x
