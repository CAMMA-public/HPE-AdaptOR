# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
from .meta_arch import GeneralizedRCNNSplitGN
from .roi_heads import KRCNNConvDeconvUpsampleHeadGN, MaskRCNNConvUpsampleHeadGN, CustomROIHeads
from .utils import convert_splitgn_model