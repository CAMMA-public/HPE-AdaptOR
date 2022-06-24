# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)
from detectron2.structures import Boxes, Instances

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA
from detectron2.modeling import GeneralizedRCNNWithTTA

__all__ = ["GeneralizedRCNNWithTTAWithKPT"]


def calc_keypoint_hflip_indices(cfg):
    from detectron2.data.detection_utils import check_metadata_consistency
    from detectron2.data.catalog import MetadataCatalog

    dataset_names = cfg.DATASETS.TEST
    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices, dtype=np.int32)


class GeneralizedRCNNWithTTAWithKPT(GeneralizedRCNNWithTTA):
    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        cfg = cfg.clone()
        cfg.defrost()
        cfg.MODEL.KEYPOINT_ON = False
        super().__init__(cfg, model, tta_mapper, batch_size)
        self.cfg.MODEL.KEYPOINT_ON = True
        self.keypoint_hflip_indices = calc_keypoint_hflip_indices(self.cfg)

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        with self._turn_off_roi_heads(["mask_on", "keypoint_on"]):
            # temporarily disable roi heads
            all_boxes, all_scores, all_classes = self._get_augmented_boxes(
                augmented_inputs, tfms
            )
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(
            all_boxes, all_scores, all_classes, orig_shape
        )

        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.KEYPOINT_ON:
            # Use the detected boxes to obtain masks
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, tfms
            )
            # run forward on the detected boxes
            outputs = self._batch_inference(augmented_inputs, augmented_instances)
            # Delete now useless variables to avoid being out of memory
            del augmented_inputs, augmented_instances
            # average the predictions
            if self.cfg.MODEL.MASK_ON:
                merged_instances.pred_masks = self._reduce_pred_masks(outputs, tfms)
            if self.cfg.MODEL.KEYPOINT_ON:
                merged_instances.pred_keypoints = self._reduce_pred_keypoints(
                    outputs, tfms
                )
            merged_instances = detector_postprocess(merged_instances, *orig_shape)
            return {"instances": merged_instances}
        else:
            return {"instances": merged_instances}

    def _reduce_pred_keypoints(self, outputs, tfms):
        # Should apply inverse transforms on masks.
        # We assume only resize & flip are used. pred_keypoints is a scale-variant
        # representation, so we handle flip as well as specially
        for output, tfm in zip(outputs, tfms):
            if output.pred_boxes.tensor.shape[0] > 0:
                original_pred_keypoints = np.concatenate(
                    [
                        [
                            tfm.inverse().apply_coords(kpt)
                            for kpt in output.pred_keypoints.cpu().numpy()
                        ]
                    ]
                )
                if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                    original_pred_keypoints = original_pred_keypoints[
                        :, self.keypoint_hflip_indices, :
                    ]
                output.pred_keypoints = torch.from_numpy(original_pred_keypoints).to(
                    output.pred_keypoints.device
                )
        all_pred_keypoints = torch.stack([o.pred_keypoints for o in outputs], dim=0)
        all_pred_keypoints = torch.mean(all_pred_keypoints, dim=0)
        return all_pred_keypoints