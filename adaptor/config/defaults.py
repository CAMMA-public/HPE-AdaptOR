# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


## solver
_C.SOLVER.USE_EMA = False
_C.SOLVER.EMA_UPDATE_PERIOD = 1
_C.SOLVER.EMA_DECAY = 0.0
_C.SOLVER.EMA_DECAY_SUP = 0.0
_C.SOLVER.IMS_PER_BATCH_LBL = 2
_C.SOLVER.IMS_PER_BATCH_UNLBL = 2


## datasets
_C.DATASETS.WEIGHTED_TRAINING_SAMPLING_WEIGHTS_LBL = (1.0,)
_C.DATASETS.WEIGHTED_TRAINING_SAMPLING_WEIGHTS_UNLBL = (1.0,)
_C.DATASETS.TRAIN_LBL = ()
_C.DATASETS.TRAIN_UNLBL = ()

# 
## dataloader settings: taken from https://github.com/facebookresearch/unbiased-teacher
_C.DATALOADER.SUP_PERCENT = 100.0 
_C.DATALOADER.RANDOM_DATA_SEED = 0  
_C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

## UDA/SSL Training settings
_C.UDA = CN()
_C.UDA.ENABLE = False
_C.UDA.TYPE = "ssl_coco" # uda_or or "ssl_coco" 
_C.UDA.BBOX_THRESHOLD = 0.7
_C.UDA.KEYPOINT_THRESHOLD = 0.06
_C.UDA.MASK_THRESHOLD = 0.5
_C.UDA.LAMBDA_U = 1.0
_C.UDA.LAMBDA_U_KEYPOINT = 1.0
_C.UDA.LAMBDA_U_MASK = 1.0
_C.UDA.RANDOM_AUGMENT_THRESH1 = 0.5
_C.UDA.RANDOM_AUGMENT_THRESH2 = 0.2
_C.UDA.APPLY_RAND_AUGMENT = False
_C.UDA.APPLY_RANDCUTOUT = True
_C.UDA.APPLY_LOW_RES_RESIZE = False
_C.UDA.APPLY_GEOMETRIC_TRANSFORM = False
_C.UDA.ADD_STRONGAUG_EXAMPLES = False
_C.UDA.TRAIN_SUP_ITER = 0

## MODEL: 
_C.MODEL.ROI_BOX_HEAD.LOSS = "CE"
_C.MODEL.USE_SPLIT_GROUP_NORM = False
_C.MODEL.SPLIT_GN_INFER_TYPE = "aux"
_C.MODEL.ROI_KEYPOINT_HEAD.NORM = ""


