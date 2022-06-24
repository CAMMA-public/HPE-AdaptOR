# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

__all__ = ["GeneralizedRCNNSplitGN"]


def split_image_list(inp_img_list: ImageList):
    sz = inp_img_list.tensor.shape[0]
    return (
        ImageList(
            inp_img_list.tensor[0 : sz // 2], inp_img_list.image_sizes[0 : sz // 2]
        ),
        ImageList(inp_img_list.tensor[sz // 2 :], inp_img_list.image_sizes[sz // 2 :]),
    )


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNSplitGN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        use_split_gn: bool = False,
    ):
        explicit_args = {
            "backbone": backbone,
            "proposal_generator": proposal_generator,
            "roi_heads": roi_heads,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "input_format": input_format,
            "vis_period": vis_period,
        }
        super(GeneralizedRCNNSplitGN, self).__init__(**explicit_args)
        self.use_split_gn = use_split_gn

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "use_split_gn": cfg.MODEL.USE_SPLIT_GROUP_NORM,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            if self.use_split_gn:
                sz = len(images)
                features_lbl = {n: f[0 : sz // 2] for n, f in features.items()}
                features_unlbl = {n: f[sz // 2 :] for n, f in features.items()}
                images_lbl, images_unlbl = split_image_list(images)
                proposals_lbl, proposal_losses_lbl = self.proposal_generator(
                    images_lbl, features_lbl, gt_instances[0 : sz // 2]
                )
                proposals_unlbl, proposal_losses_unlbl = self.proposal_generator(
                    images_unlbl, features_unlbl, gt_instances[sz // 2 :]
                )
            else:
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances
                )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        if self.use_split_gn:
            _, detector_losses_lbl = self.roi_heads(
                images_lbl, features_lbl, proposals_lbl, gt_instances[0 : sz // 2],
            )
            _, detector_losses_unlbl = self.roi_heads(
                images_unlbl, features_unlbl, proposals_unlbl, gt_instances[sz // 2 :],
            )
        else:
            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances
            )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        if self.use_split_gn:
            losses_lbl = {}
            losses_lbl.update(detector_losses_lbl)
            losses_lbl.update(proposal_losses_lbl)

            losses_unlbl = {}
            losses_unlbl.update(detector_losses_unlbl)
            losses_unlbl.update(proposal_losses_unlbl)
            return losses_lbl, losses_unlbl
        else:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

