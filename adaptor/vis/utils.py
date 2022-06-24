# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

import torch
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import subprocess
from detectron2.data.datasets.builtin_meta import (
    COCO_PERSON_KEYPOINT_NAMES,
    KEYPOINT_CONNECTION_RULES,
)
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
import pickle
from detectron2.structures import polygons_to_bitmask
from IPython.display import HTML, display

color_brightness = 150
CC = {}
COCO_COLORS_SKELETON = [
    "m",
    "m",
    "g",
    "g",
    "r",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "m",
    "g",
]
COCO_PAIRS = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

CC["r"] = (0, 0, color_brightness)
CC["g"] = (0, color_brightness, 0)
CC["b"] = (color_brightness, 0, 0)
CC["c"] = (color_brightness, color_brightness, 0)
CC["m"] = (color_brightness, 0, color_brightness)
CC["y"] = (0, color_brightness, color_brightness)
CC["w"] = (color_brightness, color_brightness, color_brightness)
CC["k"] = (0, 0, 0)

def fixed_bright_colors():
    return [
        [207, 73, 179],
        [53, 84, 209],
        [191, 197, 33],
        [221, 75, 180],
        [241, 16, 142],
        [126, 9, 231],
        [169, 52, 199],
        [229, 18, 115],
        [236, 31, 98],
        [231, 4, 80],
        [46, 194, 180],
        [35, 228, 69],
        [217, 211, 25],
        [203, 173, 34],
        [253, 10, 48],
        [50, 195, 222],
        [170, 213, 80],
        [206, 77, 13],
        [197, 178, 11],
        [204, 163, 32],
        [143, 222, 64],
        [45, 208, 109],
        [67, 185, 44],
        [91, 68, 230],
        [249, 246, 20],
        [75, 202, 201],
        [11, 202, 193],
        [40, 210, 122],
        [10, 136, 205],
        [31, 252, 54],
        [38, 230, 105],
        [193, 97, 26],
        [203, 18, 101],
        [42, 173, 94],
        [222, 45, 135],
        [33, 184, 48],
        [121, 49, 195],
        [31, 39, 226],
        [204, 48, 143],
        [220, 47, 192],
        [223, 220, 73],
        [46, 177, 170],
        [17, 245, 161],
        [159, 51, 107],
        [10, 39, 205],
        [50, 237, 101],
        [116, 35, 171],
        [213, 76, 76],
        [88, 203, 47],
        [202, 205, 14],
        [100, 233, 4],
        [227, 34, 192],
        [21, 79, 239],
        [30, 198, 36],
        [140, 38, 240],
        [97, 26, 215],
        [48, 122, 225],
        [158, 51, 196],
        [11, 212, 45],
        [190, 173, 39],
        [34, 185, 34],
        [98, 58, 219],
        [147, 233, 66],
        [44, 239, 69],
        [192, 177, 38],
        [53, 233, 53],
        [41, 222, 44],
        [228, 70, 120],
        [221, 153, 58],
        [131, 19, 222],
        [203, 27, 140],
        [170, 72, 54],
        [182, 58, 173],
        [194, 218, 84],
        [233, 34, 30],
        [100, 173, 37],
        [72, 92, 227],
        [216, 90, 183],
        [66, 215, 125],
        [183, 63, 41],
        [228, 29, 54],
        [29, 221, 125],
        [172, 12, 207],
        [20, 228, 205],
        [16, 228, 121],
        [210, 21, 198],
        [80, 135, 206],
        [196, 165, 27],
    ]

def progress_bar(value, max=100):
    """ A HTML helper function to display the progress bar
    Args:
        value ([int]): [current progress bar value]
        max (int, optional): [maximum value]. Defaults to 100.
    Returns:
        [str]: [HTML progress bar string]
    """
    return HTML(
        """
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(
            value=value, max=max
        )
    )


def images_to_video(img_folder, output_vid_file, fps=20):
    """[convert png images to video using ffmpeg]
    Args:
        img_folder ([str]): [path to images]
        output_vid_file ([str]): [Name of the output video file name]
    """
    os.makedirs(img_folder, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-threads",
        "16",
        "-i",
        f"{img_folder}/%06d.png",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]
    print(f'\nRunning "{" ".join(command)}"')
    subprocess.call(command)
    print("\nVideo generation finished")


def visualize_gt(datas, title="", iter=0):
    def bgr2rgb(im):
        b, g, r = cv2.split(im)
        return cv2.merge([r, g, b])

    def plt_imshow(im, title="", b2r=True):
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

    def vis2d(im, boxes, keypoints, colors, pairs_info, kps_names, gt_masks=None):
        # get random colors
        h, w, _ = im.shape
        for i, box in enumerate(boxes):
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                colors[i],
                2,
            )
            cv2.putText(
                im,
                str(i),
                (int(box[0]), int(box[1] + 10)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.9,
                colors[i],
                2,
                cv2.LINE_AA,
            )
        if gt_masks is not None:
            masks = []
            if "BitMasks" in str(type(gt_masks)):
                masks = [m.astype(np.uint8) for m in gt_masks.tensor.cpu().numpy()]
            elif "PolygonMasks" in str(type(gt_masks)):
                masks = [
                    polygons_to_bitmask(poly, h, w).astype(np.uint8)
                    for poly in gt_masks.polygons
                ]
            else:
                masks = retry_if_cuda_oom(paste_masks_in_image)(
                    gt_masks.to(torch.float32).cpu(),
                    torch.from_numpy(boxes),
                    (h, w),
                    0.5,
                )
                masks = [m.astype(np.uint8) for m in masks.cpu().numpy()]
            if masks:
                masks = [cv2.merge([m, m, m]) for m in masks]
                for m, c in zip(masks, colors):
                    for ii in range(3):
                        m[:, :, ii] *= c[ii]
                mask_img = masks[0]
                for m in masks[1:]:
                    mask_img = cv2.add(mask_img, m)
                im = cv2.addWeighted(im, 0.5, mask_img, 0.5, 0)

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

    for ii, data in enumerate(datas):
        for jj, dataset_dict in enumerate(data):
            if dataset_dict["instances"].has("gt_keypoints"):
                image = bgr2rgb(
                    np.transpose(dataset_dict["image"].numpy(), (1, 2, 0))
                ).astype(np.uint8)
                boxes = dataset_dict["instances"].gt_boxes.tensor.cpu().numpy()
                keypoints = dataset_dict["instances"].gt_keypoints.tensor.cpu().numpy()
                if dataset_dict["instances"].has("gt_masks"):
                    gt_masks = dataset_dict["instances"].gt_masks
                else:
                    gt_masks = None

                colors = [
                    (torch.LongTensor(3).random_(100, 255).numpy().flatten().tolist())
                    for _ in range(boxes.shape[0])
                ]
                pairs_info = KEYPOINT_CONNECTION_RULES
                kps_names = COCO_PERSON_KEYPOINT_NAMES
                im_orig = image.copy()
                fig = plt.figure(figsize=(15, 15))
                fig.add_subplot(1, 2, 1)
                plt_imshow(im_orig, "image")
                image = vis2d(
                    image, boxes, keypoints, colors, pairs_info, kps_names, gt_masks
                )
                fig.add_subplot(1, 2, 2)
                plt_imshow(image, "image-render")
                plt.show()
                # vis = np.concatenate((im_orig, image), axis=1)
                # cv2.imwrite(os.path.join( title + "_iter_" + str(iter) + "_" + str(ii) + "_" + str(jj) + ".png" , vis)


def visualize_sample(data, image_check_flip, pred_boxes, pred_keypoints):
    # print("dataset_dict:", dataset_dict)
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

    for i, dataset_dict in enumerate(data):
        if dataset_dict["instances"].has("gt_keypoints"):
            image = bgr2rgb(
                np.transpose(dataset_dict["image"].numpy(), (1, 2, 0))
            ).astype(np.uint8)
            image_flip = bgr2rgb(
                np.transpose(dataset_dict["image_flip"].numpy(), (1, 2, 0))
            ).astype(np.uint8)
            image_strong_flip = bgr2rgb(
                np.transpose(dataset_dict["image_strong_flip"].numpy(), (1, 2, 0))
            ).astype(np.uint8)
            img_check_flip = bgr2rgb(
                np.transpose(image_check_flip[i].cpu().numpy(), (1, 2, 0))
            ).astype(np.uint8)
            boxes = dataset_dict["instances"].gt_boxes.tensor.numpy()
            boxes_flip = dataset_dict["instances"].gt_boxes_flip.tensor.numpy()
            keypoints = dataset_dict["instances"].gt_keypoints.tensor.numpy()
            keypoints_flip = dataset_dict["instances"].gt_keypoints_flip.tensor.numpy()

            boxex_check_flip = pred_boxes[i].tensor.cpu().numpy()
            keypoints_check_flip = pred_keypoints[i].tensor.cpu().numpy()

            colors = [
                (torch.LongTensor(3).random_(100, 255).numpy().flatten().tolist())
                for _ in range(boxes.shape[0])
            ]
            pairs_info = KEYPOINT_CONNECTION_RULES
            kps_names = COCO_PERSON_KEYPOINT_NAMES

            # render the boxes and keypoints
            image = vis2d(image, boxes, keypoints, colors, pairs_info, kps_names)
            image_flip = vis2d(
                image_flip, boxes_flip, keypoints_flip, colors, pairs_info, kps_names
            )
            image_strong_flip = vis2d(
                image_strong_flip,
                boxes_flip,
                keypoints_flip,
                colors,
                pairs_info,
                kps_names,
            )
            img_check_flip = vis2d(
                img_check_flip,
                boxex_check_flip,
                keypoints_check_flip,
                colors,
                pairs_info,
                kps_names,
            )

            fig = plt.figure(figsize=(15, 15))
            fig.add_subplot(2, 2, 1)
            plt_imshow(image, "image")
            fig.add_subplot(2, 2, 2)
            plt_imshow(image_flip, "image_flip")
            fig.add_subplot(2, 2, 3)
            plt_imshow(image_strong_flip, "image_strong_flip")
            fig.add_subplot(2, 2, 4)
            plt_imshow(img_check_flip, "img_check_flip")
            plt.show()


def vis_2d_anns(im, anns, width, height, colors=None):
    if "mask" in anns[0]:
        masks = [a["mask"].astype(np.uint8) for a in anns]
        masks = [cv2.merge([m, m, m]) for m in masks]
        for i, m in enumerate(masks):
            c = colors[i] if colors is not None else (255, 255, 255)
            for ii in range(3):
                m[:, :, ii] *= c[ii]
        mask_img = masks[0]
        for m in masks[1:]:
            mask_img = cv2.add(mask_img, m)
        im = cv2.addWeighted(im, 0.6, mask_img, 0.4, 0)

    for i, det in enumerate(anns):
        pose = np.array(det["keypoints"]).reshape(17, 3)
        rect = det["bbox"]
        cv2.rectangle(
            im,
            (int(rect[0]), int(rect[1])),
            (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
            colors[i] if colors is not None else (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        for idx in range(len(COCO_COLORS_SKELETON)):
            pt1 = (
                int(np.clip(pose[COCO_PAIRS[idx][0] - 1, 0], 0, width)),
                int(np.clip(pose[COCO_PAIRS[idx][0] - 1, 1], 0, height)),
            )
            pt2 = (
                int(np.clip(pose[COCO_PAIRS[idx][1] - 1, 0], 0, width)),
                int(np.clip(pose[COCO_PAIRS[idx][1] - 1, 1], 0, height)),
            )
            if 0 not in pt1 + pt2:
                cv2.line(im, pt1, pt2, CC[COCO_COLORS_SKELETON[idx]], 1, cv2.LINE_AA)
        """ draw the skelton points """
        for idx_c in range(pose.shape[0]):
            pt = (
                int(np.clip(pose[idx_c, 0], 0, width)),
                int(np.clip(pose[idx_c, 1], 0, height)),
            )
            if 0 not in pt:
                cv2.circle(im, pt, 2, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(im, pt, 2, (10, 0, 125), 1, cv2.LINE_AA)

    return im
