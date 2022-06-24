#!/usr/bin/env bash
# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

PERCENTAGE_SUPERVISION=1 #2,5,10,100
CONFIG_FILE=./configs/ssl_coco/kmrcnn_R_50_FPN_3x_gn_amp_sup${PERCENTAGE_SUPERVISION}.yaml
MODEL_WEIGHTS=./models/kmrcnn_R_50_FPN_3x_gn_amp_sup${PERCENTAGE_SUPERVISION}.pth
python train_net.py \
      --eval-only \
      --config ${CONFIG_FILE} \
      MODEL.WEIGHTS ${MODEL_WEIGHTS}

