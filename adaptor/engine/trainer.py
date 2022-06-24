# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

import logging
import numpy as np
import time
import os
from typing import Dict
from collections import OrderedDict
import torch
import torch.nn as nn
from copy import deepcopy
import random
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.engine.train_loop import TrainerBase
from detectron2.utils.events import get_event_storage
from torch.nn.parallel import DataParallel, DistributedDataParallel
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks
from detectron2.evaluation.evaluator import inference_context
from detectron2.structures import Keypoints, Instances, Boxes, BitMasks
from detectron2.data import detection_utils as utils
from torch.cuda.amp import autocast
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
import pycocotools.mask as mask_util

# from detectron2.modeling import GeneralizedRCNNWithTTA

from adaptor.data import custom_train_loader
from adaptor.evaluation import GeneralizedRCNNWithTTAWithKPT
from adaptor.modeling import convert_splitgn_model
from detectron2.modeling import GeneralizedRCNNWithTTA


class ModelEMA(torch.nn.Module):
    def __init__(self, model, device=None):
        super(ModelEMA, self).__init__()
        self.ema = deepcopy(model)
        self.ema.eval()
        self.device = device
        if self.device is not None:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.ema.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, decay):
        self._update(model, update_fn=lambda e, m: decay * e + (1.0 - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class D2Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class AdaptORTrainer(DefaultTrainer):
    def __init__(self, cfg):
        self._hooks = []
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        # [TODO: integrate inside the build_model]
        if cfg.MODEL.USE_SPLIT_GROUP_NORM:
            DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
            model = convert_splitgn_model(
                model, inference_type=cfg.MODEL.SPLIT_GN_INFER_TYPE
            )
            logger.info("model with split group norm {}".format(model))

        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self._trainer = SimpleTrainer(cfg, model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            self._trainer.ema_model.ema if cfg.SOLVER.USE_EMA else model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if self.cfg.SOLVER.USE_EMA:
            self._trainer.model.load_state_dict(
                self._trainer.ema_model.ema.state_dict()
            )

        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(
                self.cfg,
                self._trainer.ema_model.ema if cfg.SOLVER.USE_EMA else self.model,
            )
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTAWithKPT(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.UDA.ENABLE:
            return custom_train_loader(cfg)
        else:
            return super().build_train_loader(cfg)


class SimpleTrainer(TrainerBase):
    def __init__(self, cfg, model, data_loader, optimizer, grad_scaler=None):
        super().__init__()
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        if self.use_amp:
            unsupported = (
                "AMPTrainer does not support single-process multi-device training!"
            )
            if isinstance(model, DistributedDataParallel):
                assert not (model.device_ids and len(model.device_ids) > 1), unsupported
            assert not isinstance(model, DataParallel), unsupported

            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
                self.grad_scaler = grad_scaler

        if isinstance(model, DistributedDataParallel):
            self.device = model.module.device
        else:
            self.device = model.device

        self.lambda_u = cfg.UDA.LAMBDA_U
        self.lambda_u_kpt = cfg.UDA.LAMBDA_U_KEYPOINT
        self.lambda_u_mask = cfg.UDA.LAMBDA_U_MASK
        self.use_ema = cfg.SOLVER.USE_EMA
        self.ema_period = cfg.SOLVER.EMA_UPDATE_PERIOD
        self.ema_decay = cfg.SOLVER.EMA_DECAY
        self.ema_decay_sup = cfg.SOLVER.EMA_DECAY_SUP
        if self.use_ema:
            self.ema_model = ModelEMA(model, self.device)

        # semi-supervised training
        self.bbox_threshold = cfg.UDA.BBOX_THRESHOLD
        self.kpt_threshold = cfg.UDA.KEYPOINT_THRESHOLD
        self.mask_threshold = cfg.UDA.MASK_THRESHOLD
        self.train_sup_iter = cfg.UDA.TRAIN_SUP_ITER
        self.add_strong_aug_examples = cfg.UDA.ADD_STRONGAUG_EXAMPLES
        self.with_split_gn = cfg.MODEL.USE_SPLIT_GROUP_NORM
        self.is_dist = (
            hasattr(self.ema_model.ema, "module")
            if self.use_ema
            else hasattr(model, "module")
        )
        self._cpu_device = torch.device("cpu")
        self.flip_keypoint2d = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        model.train()
        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        self.optimizer = optimizer

        self.idxs_bb = torch.tensor(
            np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten(), dtype=torch.int64
        ).to(self.device)

    def run_inference(self, data):
        if self.use_ema:
            if self.is_dist:
                pred_model = self.ema_model.ema.module
            else:
                pred_model = self.ema_model.ema
        else:
            if self.is_dist:
                pred_model = self.model.module
            else:
                pred_model = self.model
        with inference_context(pred_model), torch.no_grad():
            if self.use_amp:
                with autocast():
                    output = pred_model.inference(data, do_postprocess=False)
            else:
                output = pred_model.inference(data, do_postprocess=False)

            keep_preds = [o.scores > self.bbox_threshold for o in output]
            pred_boxes = [o.pred_boxes[k] for o, k in zip(output, keep_preds)]
            if output[0].has("pred_keypoints"):
                pred_keypoints = [
                    o.pred_keypoints[k] for o, k in zip(output, keep_preds)
                ]
                vis = [
                    (k[..., -1] > self.kpt_threshold)[..., None].repeat(1, 1, 3)
                    for k in pred_keypoints
                ]
                for p in pred_keypoints:
                    p[..., -1] = 2.0
                pred_keypoints = [
                    Keypoints(torch.where(v, k, torch.zeros_like(k)))
                    for v, k in zip(vis, pred_keypoints)
                ]
            else:
                pred_keypoints = None
            if output[0].has("pred_masks"):
                pred_masks = [o.pred_masks[k] for o, k in zip(output, keep_preds)]
            else:
                pred_masks = None

            image_sizes = [o.image_size for o in output]

        return pred_boxes, pred_keypoints, pred_masks, image_sizes

    def flip_boxes(self, pred_box, size):
        box = pred_box.tensor
        # box flip
        coords = box.view(-1, 4)[:, self.idxs_bb].view(-1, 2)
        coords[:, 0] = size[-1] - coords[:, 0]
        coords = coords.view(-1, 4, 2)
        minxy = coords.min(dim=1).values
        maxxy = coords.max(dim=1).values
        box = torch.cat((minxy, maxxy), dim=1)
        sz = torch.tensor(size + size).to(box.device).to(box.dtype)
        box = Boxes(torch.min(box, torch.flip(sz, [0])[None]))
        return box

    def flip_keypoints(self, pred_keypoints, size):
        kps = pred_keypoints.tensor
        kps[..., 0] = size[-1] - kps[..., 0]
        kps = Keypoints(kps[:, self.flip_keypoint2d, :])
        return kps

    def get_pseudo_labels(self, data_unlbl_weak, data_unlbl_strong):
        out_data = []
        pred_boxes, pred_keypoints, pred_masks, image_sizes = self.run_inference(
            data_unlbl_weak
        )
        for i, d in enumerate(data_unlbl_weak):
            # if atleast one bounding box is found after the inference
            if pred_boxes[i].tensor.shape[0] >= 1:
                # if np.random.random() < 0.5:
                d["image"] = data_unlbl_strong[i]["image"].clone()
                flip_enable = d["flip_enable"]
                new_size = (d["image"].shape[1], d["image"].shape[2])
                target = Instances(new_size)
                s_x, s_y = (
                    target.image_size[1] / image_sizes[i][1],
                    target.image_size[0] / image_sizes[i][0],
                )
                # print("scale", s_x, s_y, " flip", flip_enable)
                pred_boxes[i].scale(s_x, s_y)
                pred_boxes[i].clip(target.image_size)
                if pred_keypoints is not None:
                    pred_keypoints[i].tensor[..., 0] *= s_x
                    pred_keypoints[i].tensor[..., 1] *= s_y
                if flip_enable:
                    target.gt_boxes = self.flip_boxes(pred_boxes[i], new_size)
                    if pred_keypoints is not None:
                        target.gt_keypoints = self.flip_keypoints(
                            pred_keypoints[i], new_size
                        )
                    if pred_masks is not None:
                        target.gt_masks = (
                            pred_masks[i].flip(dims=[3]) >= self.mask_threshold
                        ).to(dtype=torch.bool)[:, 0, :, :]
                else:
                    target.gt_boxes = pred_boxes[i]
                    if pred_keypoints is not None:
                        target.gt_keypoints = pred_keypoints[i]
                    if pred_masks is not None:
                        target.gt_masks = (pred_masks[i] >= self.mask_threshold).to(
                            dtype=torch.bool
                        )[:, 0, :, :]
                target.gt_classes = torch.zeros(
                    pred_boxes[i].tensor.shape[0], dtype=torch.int64
                ).to(pred_boxes[i].tensor.device)

                d["instances"] = target
                out_data.append(d)
        return out_data

    def compute_losses(self, data_lbl, data_lbl_pseudo):
        if self.with_split_gn:
            if len(data_lbl) == len(data_lbl_pseudo):
                loss_dict_lbl, loss_dict_unlbl = self.model(data_lbl + data_lbl_pseudo)
                loss_dict_unlbl = dict(
                    (k + "_unlabel", self.lambda_u_kpt * v)
                    if "keypoint" in k
                    else (k + "_unlabel", self.lambda_u_mask * v)
                    if "mask" in k
                    else (k + "_unlabel", self.lambda_u * v)
                    for (k, v) in loss_dict_unlbl.items()
                )
            else:
                loss_dict_lbl, loss_dict_unlbl = self.model(data_lbl + data_lbl)
                loss_dict_unlbl = dict(
                    (k + "_unlabel", 1.0 * v) for (k, v) in loss_dict_unlbl.items()
                )
            losses = sum(loss_dict_lbl.values()) + sum(loss_dict_unlbl.values())
        else:
            loss_dict_lbl = self.model(data_lbl)
            if len(data_lbl_pseudo) > 0:
                loss_dict_unlbl = self.model(data_lbl_pseudo)
                loss_dict_unlbl = dict(
                    (k + "_unlabel", self.lambda_u_kpt * v)
                    if "keypoint" in k
                    else (k + "_unlabel", self.lambda_u_mask * v)
                    if "mask" in k
                    else (k + "_unlabel", self.lambda_u * v)
                    for (k, v) in loss_dict_unlbl.items()
                )
                losses = sum(loss_dict_lbl.values()) + sum(loss_dict_unlbl.values())
            else:
                loss_dict_unlbl = dict(
                    (k + "_unlabel", torch.tensor([0.0]))
                    for (k, v) in loss_dict_lbl.items()
                )
                losses = sum(loss_dict_lbl.values())

        loss_dict = dict(loss_dict_lbl, **loss_dict_unlbl)

        return losses, loss_dict

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_lbl_weak, data_lbl_strong, data_unlbl_weak, data_unlbl_strong = data
        data_time = time.perf_counter() - start

        if self.iter < self.train_sup_iter:
            if self.use_ema:
                self.ema_model.update(self.model, decay=self.ema_decay_sup)

            if self.with_split_gn:
                # for the training of baseline methods(pseudo-label, data-distillation, orpose)
                assert len(data_unlbl_strong) == len(data_lbl_weak) + len(
                    data_lbl_strong
                )
                data_lbl = data_lbl_weak + data_lbl_strong + data_unlbl_strong
            else:
                if self.add_strong_aug_examples:
                    data_lbl = data_lbl_weak
                    data_lbl.extend(data_lbl_strong)
                else:
                    data_lbl = data_lbl_weak

            if self.use_amp:
                with autocast():
                    loss_dict = self.model(data_lbl)
                    losses = sum(loss_dict.values())
            else:
                loss_dict = self.model(data_lbl)
                losses = sum(loss_dict.values())
        else:
            """
            EMA update
            """
            if (self.iter - self.train_sup_iter) % self.ema_period == 0 or (
                self.iter == self.max_iter - 1
            ):
                if self.use_ema:
                    self.ema_model.update(self.model, decay=self.ema_decay)
            loss_dict = {}

            data_lbl_pseudo = self.get_pseudo_labels(data_unlbl_weak, data_unlbl_strong)
            data_lbl = data_lbl_weak
            data_lbl.extend(data_lbl_strong)

            if self.use_amp:
                with autocast():
                    losses, loss_dict = self.compute_losses(data_lbl, data_lbl_pseudo)
            else:
                losses, loss_dict = self.compute_losses(data_lbl, data_lbl_pseudo)

        if self.use_amp:
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()
            self._write_metrics(loss_dict, data_time)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.zero_grad()
            losses.backward()
            self._write_metrics(loss_dict, data_time)
            self.optimizer.step()

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)
