# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

from .data import (
    custom_train_loader,
    WeightedTrainingSampler,
    DatasetMapperWeakStrong,
    RandAugment,
)
from .engine import AdaptORTrainer
from .evaluation import test_time_augmentation_with_kpt
from .modeling import (
    GeneralizedRCNNSplitGN,
    KRCNNConvDeconvUpsampleHeadGN,
    MaskRCNNConvUpsampleHeadGN,
    CustomROIHeads,
    KRCNNConvDeconvUpsampleHeadGN,
    GeneralizedRCNNSplitGN,
    convert_splitgn_model,
)

from .vis import vis_2d_anns, fixed_bright_colors, progress_bar, images_to_video