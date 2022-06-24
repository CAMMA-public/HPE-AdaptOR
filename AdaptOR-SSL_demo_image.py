from logging import debug
import torch
import detectron2
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from PIL import Image
import gradio as gr

from adaptor.config import get_cfg
from adaptor.vis import vis_2d_anns, fixed_bright_colors

BBOX_THRESHOLD, KPT_THRESHOLD, MASK_THRESHOLD = 0.7, 0.1, 0.5
IMG_RESIZE_SHAPE = 640
DO_RESIZE = True
IMG_FORMAT = "RGB"
CFG_FILE = "./configs/base/Base_kmrcnn_R_50_FPN_3x_gn_amp_4gpus.yaml"
predictors = {}
MODEL_WEIGHTS = {
    "1%": "./models/kmrcnn_R_50_FPN_3x_gn_amp_sup1.pth",
    "2%": "./models/kmrcnn_R_50_FPN_3x_gn_amp_sup2.pth",
    "5%": "./models/kmrcnn_R_50_FPN_3x_gn_amp_sup5.pth",
    "10%": "./models/kmrcnn_R_50_FPN_3x_gn_amp_sup10.pth",
    "100%": "./models/kmrcnn_R_50_FPN_3x_gn_amp_sup100.pth",
}
title = "AdaptOR-SSL Image Demo"
description = "Image demo for AdaptOR-SSL. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/'></a> | <a href='https://github.com/CAMMA-public/AdaptOR'>Github Repo</a></p>"


for MODEL_TYPE, WEIGHT in MODEL_WEIGHTS.items():
    cfg = get_cfg()
    cfg.merge_from_file(CFG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = BBOX_THRESHOLD
    cfg.MODEL.WEIGHTS = WEIGHT
    predictors[MODEL_TYPE] = DefaultPredictor(cfg)

colors = fixed_bright_colors()


def adaptor_ssl_image(img, ssl_model_type):
    assert ssl_model_type in ["1%", "2%", "5%", "10%", "100%"]
    predictor = predictors[ssl_model_type]

    det_anns = []
    img = utils.read_image(img.name, format=IMG_FORMAT)
    if DO_RESIZE:
        img, _ = T.apply_transform_gens(
            [T.ResizeShortestEdge(short_edge_length=IMG_RESIZE_SHAPE)], img
        )
    outputs = predictor(img)
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    pred_keypoints = outputs["instances"].pred_keypoints.cpu()
    pred_masks = (
        (outputs["instances"].pred_masks >= MASK_THRESHOLD)
        .to(dtype=torch.bool)
        .cpu()
        .numpy()
    )
    vis = (pred_keypoints[..., -1] > KPT_THRESHOLD)[..., None].repeat(1, 1, 3)
    pred_keypoints = torch.where(vis, pred_keypoints, torch.zeros_like(pred_keypoints))
    h, w = outputs["instances"].image_size
    for box, kpt, mask in zip(pred_boxes, pred_keypoints, pred_masks):
        det_anns.append(
            {
                "keypoints": kpt.numpy(),
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "mask": mask,
            }
        )
    det_anns = sorted(det_anns, key=lambda k: k["bbox"][0])
    img = vis_2d_anns(img, det_anns, w, h, colors=colors)
    image_pil = Image.fromarray(img)
    return image_pil


gr.Interface(
    adaptor_ssl_image,
    [
        gr.inputs.Image(type="file", label="Input Image"),
        gr.inputs.Radio(
            ["1%", "2%", "5%", "10%", "100%"], label="SSL Model with supervision"
        ),
    ],
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
        ["utils_data/coco_val2017_000000410456.jpg", "1%"],
        ["utils_data/coco_val2017_000000410456.jpg", "2%"],
        ["utils_data/coco_val2017_000000410456.jpg", "5%"],
        ["utils_data/coco_val2017_000000410456.jpg", "10%"],
        ["utils_data/coco_val2017_000000410456.jpg", "100%"],
    ],
    server_name="0.0.0.0",
).launch(debug=True)
