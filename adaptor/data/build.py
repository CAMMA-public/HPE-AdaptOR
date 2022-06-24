# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
import numpy as np
import json
import logging
import torch
from detectron2.utils.comm import get_world_size

import operator
from detectron2.data.samplers import (
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from .samplers import WeightedTrainingSampler
from detectron2.data.build import get_detection_dataset_dicts

from detectron2.data.build import worker_init_reset_seed
from detectron2.data.common import (
    AspectRatioGroupedDataset,
    DatasetFromList,
    MapDataset,
)

logger = logging.getLogger(__name__)
from adaptor.data.dataset_mapper_weak_strong import DatasetMapperWeakStrong

__all__ = [
    "custom_train_loader",
]

# modified from: https://github.com/facebookresearch/unbiased-teacher/blob/main/ubteacher/data/build.py
def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    # read from pre-generated data seed
    with open(random_data_seed_path) as COCO_sup_file:
        coco_random_idx = json.load(COCO_sup_file)

    labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    # assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])
    return label_dicts, unlabel_dicts


# assuming two dataset (coco and mvor)
def custom_train_loader(cfg):
    logger = logging.getLogger(__name__)
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    assert cfg.UDA.ENABLE == True
    mapper_unlbl = DatasetMapperWeakStrong(cfg, True, is_labeled=False)
    mapper_lbl = DatasetMapperWeakStrong(cfg, True, is_labeled=True)
    if cfg.UDA.TYPE == "ssl_coco":
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        dataset_lbl, dataset_unlbl = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )
        logger.info(
            "Num of labeled images : {}, Num of labeled instances : {}".format(
                len(dataset_lbl), sum([len(d["annotations"]) for d in dataset_lbl])
            )
        )
        logger.info("Num of unlabeled images : {}".format(len(dataset_unlbl)))
    elif cfg.UDA.TYPE == "uda_or":
        dataset_unlbl = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLBL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        dataset_lbl = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LBL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:
        assert False, "please check:UDA.TYPE should either contain 'ssl_coco' or 'uda_or' "

    if sampler_name == "TrainingSampler":
        sampler_unlbl = TrainingSampler(len(dataset_unlbl))
        sampler_lbl = TrainingSampler(len(dataset_lbl))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors_unlbl = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_unlbl, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        repeat_factors_lbl = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_lbl, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler_unlbl = RepeatFactorTrainingSampler(repeat_factors_unlbl)
        sampler_lbl = RepeatFactorTrainingSampler(repeat_factors_lbl)
    elif sampler_name == "WeightedTrainingSampler":
        if len(cfg.DATASETS.TRAIN_UNLBL) > 1:
            hist = {d: 0 for d in cfg.DATASETS.TRAIN_UNLBL}
            for d in dataset_unlbl:
                hist[d["dataset"]] += 1
            sample_weights = {
                t: s
                for t, s in zip(
                    cfg.DATASETS.TRAIN_UNLBL,
                    cfg.DATASETS.WEIGHTED_TRAINING_SAMPLING_WEIGHTS_UNLBL,
                )
            }
            weights = torch.as_tensor(
                [
                    sample_weights[d["dataset"]] / hist[d["dataset"]]
                    for d in dataset_unlbl
                ],
                dtype=torch.double,
            )
            sampler_unlbl = WeightedTrainingSampler(len(dataset_unlbl), weights)
        else:
            sampler_unlbl = TrainingSampler(len(dataset_unlbl))
        if len(cfg.DATASETS.TRAIN_LBL) > 1:
            hist = {d: 0 for d in cfg.DATASETS.TRAIN_LBL}
            for d in dataset_lbl:
                hist[d["dataset"]] += 1
            sample_weights = {
                t: s
                for t, s in zip(
                    cfg.DATASETS.TRAIN_LBL,
                    cfg.DATASETS.WEIGHTED_TRAINING_SAMPLING_WEIGHTS_LBL,
                )
            }
            weights = torch.as_tensor(
                [
                    sample_weights[d["dataset"]] / hist[d["dataset"]]
                    for d in dataset_lbl
                ],
                dtype=torch.double,
            )
            sampler_lbl = WeightedTrainingSampler(len(dataset_lbl), weights)
        else:
            sampler_lbl = TrainingSampler(len(dataset_lbl))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    assert isinstance(sampler_unlbl, torch.utils.data.sampler.Sampler)
    assert isinstance(sampler_lbl, torch.utils.data.sampler.Sampler)

    # convert datadict to binary
    if isinstance(dataset_unlbl, list):
        dataset_unlbl = DatasetFromList(dataset_unlbl, copy=False)

    if isinstance(dataset_lbl, list):
        dataset_lbl = DatasetFromList(dataset_lbl, copy=False)

    # use the mapper to map the dataset
    dataset_unlbl = MapDataset(dataset_unlbl, mapper_unlbl)
    dataset_lbl = MapDataset(dataset_lbl, mapper_lbl)

    aspect_ratio_grouping = cfg.DATALOADER.ASPECT_RATIO_GROUPING
    num_workers = cfg.DATALOADER.NUM_WORKERS
    total_batch_size_unlbl = cfg.SOLVER.IMS_PER_BATCH_UNLBL
    total_batch_size_lbl = cfg.SOLVER.IMS_PER_BATCH_LBL

    return build_semisup_batch_loader(
        (dataset_lbl, dataset_unlbl),
        (sampler_lbl, sampler_unlbl),
        total_batch_size_lbl,
        total_batch_size_unlbl,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


# taken from:https://github.com/facebookresearch/unbiased-teacher/blob/main/ubteacher/data/build.py
def build_semisup_batch_loader(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDataset(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")


# taken from:https://github.com/facebookresearch/unbiased-teacher/blob/main/ubteacher/data/build.py
class AspectRatioGroupedSemiSupDataset(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (int):
        """

        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = [[] for _ in range(2)]
        self._label_buckets_key = [[] for _ in range(2)]
        self._unlabel_buckets = [[] for _ in range(2)]
        self._unlabel_buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            # d[0] is with strong augmentation, d[1] is with weak augmentation

            # because we are grouping images with their aspect ratio
            # label and unlabel buckets might not have the same number of data
            # i.e., one could reach batch_size, while the other is still not
            if len(label_bucket) != self.batch_size_label:
                w, h = d_label[0]["width"], d_label[0]["height"]
                label_bucket_id = 0 if w > h else 1
                label_bucket = self._label_buckets[label_bucket_id]
                label_bucket.append(d_label[0])
                label_buckets_key = self._label_buckets_key[label_bucket_id]
                label_buckets_key.append(d_label[1])

            if len(unlabel_bucket) != self.batch_size_unlabel:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
                unlabel_bucket.append(d_unlabel[0])
                unlabel_buckets_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_buckets_key.append(d_unlabel[1])

            # yield the batch of data until all buckets are full
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                yield (
                    label_bucket[:],
                    label_buckets_key[:],
                    unlabel_bucket[:],
                    unlabel_buckets_key[:],
                )
                del label_bucket[:]
                del label_buckets_key[:]
                del unlabel_bucket[:]
                del unlabel_buckets_key[:]
