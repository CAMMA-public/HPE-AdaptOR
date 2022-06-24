# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
from torch import nn
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple
from detectron2.modeling.poolers import ROIPooler 

from .fast_rcnn_focal import FastRCNNFocaltLossOutputLayers

__all__ = ["CustomROIHeads"]


@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs
    ):
        explicit_args = {
            "box_in_features": box_in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "mask_in_features": mask_in_features,
            "mask_pooler": mask_pooler,
            "mask_head": mask_head,
            "keypoint_in_features": keypoint_in_features,
            "keypoint_pooler": keypoint_pooler,
            "keypoint_head": keypoint_head,
            "train_on_pred_boxes": train_on_pred_boxes,
        }
        explicit_args.update(kwargs)
        super(CustomROIHeads, self).__init__(**explicit_args)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if cfg.MODEL.ROI_BOX_HEAD.LOSS == "FocalLoss":
            ret["box_predictor"] = FastRCNNFocaltLossOutputLayers(cfg, ret["box_head"].output_shape)
        return ret