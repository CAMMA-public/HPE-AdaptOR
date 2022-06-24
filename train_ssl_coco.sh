#!/usr/bin/env bash
# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

PERCENTAGE_SUPERVISION=1 #2,5,10,100
CONFIG_FILE=./configs/ssl_coco/kmrcnn_R_50_FPN_3x_gn_amp_sup${PERCENTAGE_SUPERVISION}.yaml
python train_net.py \
      --num-gpus 4 \
      --config ${CONFIG_FILE} \
