# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

import copy
import logging
import numpy as np
import cv2
from typing import List, Optional, Union
import torch
from detectron2.config import configurable
import random
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import Transform, TransformList
from detectron2.data.transforms import ResizeTransform
from detectron2.data.detection_utils import (
    _apply_exif_orientation,
    convert_PIL_to_numpy,
)
from PIL import Image
from detectron2.data.transforms.augmentation import _get_aug_input_args
from .randaugment import RandAugment, Cutout

__all__ = ["DatasetMapperSemiSupervised"]


class RandomAugumnetCutOut:
    """ Resize image to low res and transform back"""

    def __init__(self, apply_cutout=True):
        self.random_transform = RandAugment()
        self.apply_cutout = apply_cutout
        if self.apply_cutout:
            self.cutout = Cutout()

    def __call__(self, img):
        img = self.random_transform(img)
        if self.apply_cutout:
            num = np.random.randint(2, 16)
            for _ in range(num):
                img = self.cutout(img, np.random.randint(40, 80))
        return img


class LowResResize(T.Augmentation):
    """ Resize image to low res and transform back"""

    def __init__(self, cfg, is_train, interp=Image.BILINEAR, is_ds_mvor=True):
        self.is_train = is_train
        self.interp = interp
        self.is_ds_mvor = is_ds_mvor

    def _get_scale(self, w, dataset_name):
        if "mvor" in dataset_name or "tum" in dataset_name:
            scale = random.uniform(1.0, 12.0)
        else:
            scale = random.uniform(1.0, 4.0)
        return scale

    def get_transform(self, image, dataset_name):
        h, w, _ = image.shape
        scale = self._get_scale(w, dataset_name)
        new_h, new_w = int(h / scale), int(w / scale)
        return TransformList(
            [
                ResizeTransform(h, w, new_h, new_w, self.interp),
                ResizeTransform(new_h, new_w, h, w, self.interp),
            ]
        )

    def __call__(self, aug_input, dataset_name) -> Transform:
        args = _get_aug_input_args(self, aug_input)
        tfm = self.get_transform(*args)
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform! "
            "Got {type(tfm)} instead."
        )
        aug_input.transform(tfm)
        return tfm


def build_augmentation(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        apply_cutout = cfg.UDA.APPLY_RANDCUTOUT
        aug_main_noflip = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        aug_main = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        aug_main.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
        low_res_resize = LowResResize(cfg, is_train)
        rand_augment = RandomAugumnetCutOut(apply_cutout=apply_cutout)
        return aug_main, aug_main_noflip, rand_augment, low_res_resize
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_main = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        return aug_main


class DatasetMapperWeakStrong:
    def __init__(
        self, cfg, is_train: bool = True, is_labeled: bool = True,
    ):
        aug_main, aug_main_noflip, rand_augment, low_res_resize = build_augmentation(
            cfg, is_train
        )
        if cfg.INPUT.CROP.ENABLED and is_train:
            aug_main.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT
        self.use_keypoint = cfg.MODEL.KEYPOINT_ON
        self.recompute_boxes = recompute_boxes
        self.apply_rand_augment = cfg.UDA.APPLY_RAND_AUGMENT
        self.apply_low_res_resize = cfg.UDA.APPLY_LOW_RES_RESIZE
        self.apply_geometric_transform = cfg.UDA.APPLY_GEOMETRIC_TRANSFORM
        self.rand_aug_thresh1 = cfg.UDA.RANDOM_AUGMENT_THRESH1
        self.rand_aug_thresh2 = cfg.UDA.RANDOM_AUGMENT_THRESH2
        self.is_labeled = is_labeled

        self.aug_main = T.AugmentationList(aug_main)
        self.aug_main_noflip = T.AugmentationList(aug_main_noflip)
        self.aug_lowres_resize = low_res_resize
        self.aug_random_augment = rand_augment
        if cfg.MODEL.KEYPOINT_ON:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None
        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        if recompute_boxes:
            assert self.use_instance_mask, "recompute_boxes requires instance masks"

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {aug_main}")

    def _get_dataset_name(self, file_name):
        if "mvor_dataset" in file_name:
            dataset_name = "mvor_dataset"
        elif "TUM_OR" in file_name:
            dataset_name = "tum_or"
        elif "coco" in file_name:
            dataset_name = "coco"
        else:
            dataset_name = "gen"
        return dataset_name

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        dataset_name = self._get_dataset_name(dataset_dict["file_name"])

        if self.apply_rand_augment and self.apply_low_res_resize:
            if np.random.random() < self.rand_aug_thresh1:
                image_strong = image.copy()
                if self.image_format == "RGB":
                    image_strong = self.aug_random_augment(
                        Image.fromarray(image_strong)
                    )
                    image_strong = np.asarray(image_strong)
                    if np.random.random() < self.rand_aug_thresh2:
                        aug_input = T.AugInput(image_strong)
                        aug_input.dataset_name = dataset_name
                        _ = self.aug_lowres_resize(aug_input, dataset_name)
                        image_strong = aug_input.image
                else:
                    image_strong = self.aug_random_augment(
                        Image.fromarray(image_strong[:, :, ::-1])
                    )
                    image_strong = np.asarray(image_strong)[:, :, ::-1]
                    if np.random.random() < self.rand_aug_thresh2:
                        aug_input = T.AugInput(image_strong)
                        aug_input.dataset_name = dataset_name
                        _ = self.aug_lowres_resize(aug_input, dataset_name)
                        image_strong = aug_input.image
            else:
                aug_input = T.AugInput(image)
                aug_input.dataset_name = dataset_name
                _ = self.aug_lowres_resize(aug_input, dataset_name)
                image_strong = aug_input.image
        else:
            if self.apply_rand_augment:
                image_strong = image.copy()
                if self.image_format == "RGB":
                    image_strong = self.aug_random_augment(
                        Image.fromarray(image_strong)
                    )
                    image_strong = np.asarray(image_strong)
                else:
                    image_strong = self.aug_random_augment(
                        Image.fromarray(image_strong[:, :, ::-1])
                    )
                    image_strong = np.asarray(image_strong)[:, :, ::-1]
            elif self.apply_low_res_resize:
                aug_input = T.AugInput(image)
                aug_input.dataset_name = dataset_name
                _ = self.aug_lowres_resize(aug_input, dataset_name)
                image_strong = aug_input.image
            else:
                image_strong = image.copy()

        if self.apply_geometric_transform:
            aug_input = T.AugInput(image)
            transforms = self.aug_main(aug_input)
            image = aug_input.image
            aug_input = T.AugInput(image_strong)
            geom_transforms = self.aug_main(aug_input)
            image_strong = aug_input.image
            hflip_weak = (
                sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2
                == 1
            )
            hflip_strong = (
                sum(isinstance(t, T.HFlipTransform) for t in geom_transforms.transforms)
                % 2
                == 1
            )
            flip_enable = (
                hflip_weak ^ hflip_strong
            )  # xor (if atleat one operand is true)
        else:
            aug_input = T.AugInput(image)
            transforms = self.aug_main_noflip(aug_input)
            image = aug_input.image
            aug_input = T.AugInput(image_strong)
            for tfm in transforms:
                aug_input.transform(tfm)
            image_strong = aug_input.image
            flip_enable = False
            geom_transforms = transforms

        dataset_dict_weak = dataset_dict
        dataset_dict_strong = copy.deepcopy(dataset_dict_weak)
        dataset_dict_weak["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict_weak["flip_enable"] = flip_enable
        dataset_dict_strong["flip_enable"] = flip_enable

        dataset_dict_strong["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong.transpose(2, 0, 1))
        )

        # if not self.is_labeled:
        #    # USER: Modify this if you want to keep them for some reason.
        #    dataset_dict_weak.pop("annotations", None)
        #    dataset_dict_weak.pop("sem_seg_file_name", None)
        #    dataset_dict_strong.pop("annotations", None)
        #    dataset_dict_strong.pop("sem_seg_file_name", None)
        #    return dataset_dict_weak, dataset_dict_strong

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            # Annotations for the weak image
            for anno in dataset_dict_weak["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)
            image_shape = image.shape[:2]
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict_weak.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict_weak["instances"] = utils.filter_empty_instances(instances)

            # Annotations for the strong image
            if self.apply_geometric_transform:
                for anno in dataset_dict_strong["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)
                image_shape = image_strong.shape[:2]
                annos = [
                    utils.transform_instance_annotations(
                        obj,
                        geom_transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    )
                    for obj in dataset_dict_strong.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )
                if self.recompute_boxes:
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict_strong["instances"] = utils.filter_empty_instances(
                    instances
                )
            else:
                dataset_dict_strong["instances"] = copy.deepcopy(
                    dataset_dict_weak["instances"]
                )
        # self._visualize_sample([dataset_dict_weak, dataset_dict_strong])
        return dataset_dict_weak, dataset_dict_strong

    def _visualize_sample(self, dataset_dicts):
        # print("dataset_dict:", dataset_dict)
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        from detectron2.data.datasets.builtin_meta import (
            COCO_PERSON_KEYPOINT_NAMES,
            KEYPOINT_CONNECTION_RULES,
        )

        def bgr2rgb(im):
            b, g, r = cv2.split(im)
            return cv2.merge([r, g, b])

        def plt_imshow(im, title="", b2r=False):
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplots_adjust(
                left=0.01, bottom=0.01, right=1.0, top=1.0, wspace=0.01, hspace=0.01
            )
            if b2r:
                b, g, r = cv2.split(im)
                plt.imshow(cv2.merge([r, g, b]))
            else:
                plt.imshow(im)
            plt.title(title)

        def vis2d(im, boxes, keypoints, colors, pairs_info, kps_names):
            # get random colors
            for i, box in enumerate(boxes):
                cv2.rectangle(
                    im,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    colors[i],
                    2,
                )
            for kps in keypoints:
                for kp1_id, kp2_id, c in pairs_info:
                    kp1_id, kp2_id = kps_names.index(kp1_id), kps_names.index(kp2_id)
                    pt1 = tuple(kps[kp1_id, :].astype(int)[0:2])
                    pt2 = tuple(kps[kp2_id, :].astype(int)[0:2])
                    if 0 not in pt1 + pt2:
                        cv2.line(im, pt1, pt2, c, 2, cv2.LINE_AA)

            for i, kps in enumerate(keypoints):
                for pt in kps:
                    ptp = (int(pt[0]), int(pt[1]))
                    cv2.circle(im, ptp, 1, (0, 0, 0), 1, cv2.LINE_AA)

            return im

        if dataset_dicts[0]["instances"].has("gt_keypoints"):
            image = bgr2rgb(
                np.transpose(dataset_dicts[0]["image"].numpy(), (1, 2, 0))
            ).astype(np.uint8)
            image_strong = bgr2rgb(
                np.transpose(dataset_dicts[1]["image"].numpy(), (1, 2, 0))
            ).astype(np.uint8)
            boxes = dataset_dicts[0]["instances"].gt_boxes.tensor.numpy()
            boxes_strong = dataset_dicts[1]["instances"].gt_boxes.tensor.numpy()
            keypoints = dataset_dicts[0]["instances"].gt_keypoints.tensor.numpy()
            keypoints_strong = dataset_dicts[1]["instances"].gt_keypoints.tensor.numpy()
            print("image weak", image.shape)
            print("image strong", image_strong.shape, "\n")

            colors = [
                (torch.LongTensor(3).random_(100, 255).numpy().flatten().tolist())
                for _ in range(boxes.shape[0])
            ]
            pairs_info = KEYPOINT_CONNECTION_RULES
            kps_names = COCO_PERSON_KEYPOINT_NAMES

            # render the boxes and keypoints
            image = vis2d(image, boxes, keypoints, colors, pairs_info, kps_names)
            image_strong = vis2d(
                image_strong,
                boxes_strong,
                keypoints_strong,
                colors,
                pairs_info,
                kps_names,
            )
            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(1, 2, 1)
            plt_imshow(image, "image", b2r=True)
            fig.add_subplot(1, 2, 2)
            plt_imshow(image_strong, "image_strong", b2r=True)
            plt.show()
