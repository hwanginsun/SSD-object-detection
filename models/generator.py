"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical


class PriorBoxes:
    """
    Default Box Configuration Class
    """
    bbox_df = pd.DataFrame()

    def __init__(self, strides, scales, ratios):
        self.strides = strides
        self.scales = scales
        self.ratios = ratios
        self.setup()

    def generate(self, image_shape):
        height, width = image_shape[:2]
        centers = []
        for idx, row in self.bbox_df.iterrows():
            stride, box_width, box_height = row.stride, row.w, row.h
            ys, xs = np.mgrid[0:height:stride, 0:width:stride]
            box_width = np.ones_like(xs) * box_width
            box_height = np.ones_like(ys) * box_height
            center_xs = stride // 2 + xs
            center_ys = stride // 2 + ys

            block_centers = np.stack((center_xs, center_ys,
                                      box_width, box_height),
                                     axis=-1)
            block_centers = block_centers.reshape(-1, 4)
            centers.append(block_centers)
        return np.concatenate(centers, axis=0)

    def setup(self):
        bbox_df = pd.DataFrame(columns=['stride', 'scale', 'ratio', 'w', 'h'])
        for scale, stride in zip(self.scales, self.strides):
            for ratio in self.ratios:
                w = np.round(scale * np.sqrt(ratio)).astype(np.int)
                h = np.round(scale / np.sqrt(ratio)).astype(np.int)
                bbox_df.loc[len(bbox_df) + 1] = [stride, scale, ratio, w, h]

        bbox_df.stride = bbox_df.stride.astype(np.int)
        bbox_df.scale = bbox_df.scale.astype(np.int)
        bbox_df.ratio = bbox_df.ratio.astype(np.float)
        bbox_df.w = bbox_df.w.astype(np.int)
        bbox_df.h = bbox_df.h.astype(np.int)
        self.bbox_df = bbox_df


class DetectionGenerator(Sequence):
    'Generates Localization dataset for Keras'
    def __init__(self, dataset, prior:PriorBoxes, batch_size=32,
                 iou_threshold=0.5,
                 shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.prior = prior
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.shuffle = shuffle
        self.num_classes = dataset.num_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        images, ground_truths = self.dataset[self.batch_size * index:
                                             self.batch_size * (index + 1)]
        pr_boxes = self.prior.generate(images.shape[1:])

        y_true = []
        for index, gt_df in ground_truths.groupby('image_index'):
            gt_boxes = gt_df.iloc[:, 1:5].values
            gt_labels = gt_df.iloc[:, -1].values
            iou = calculate_iou(gt_boxes, pr_boxes)

            match_indices = np.argwhere(iou >= self.iou_threshold)

            y_true_clf = np.ones((pr_boxes.shape[0])) * self.num_classes  # Background로 일단 채움
            y_true_clf[match_indices[:, 1]] = gt_labels[match_indices[:, 0]]

            y_true_clf = to_categorical(y_true_clf, num_classes=self.num_classes + 1)  # One-Hot Encoding
            y_true_loc = np.zeros((pr_boxes.shape[0], 4))
            g_cx, g_cy, g_w, g_h = gt_boxes[match_indices[:, 0]].transpose()
            p_cx, p_cy, p_w, p_h = pr_boxes[match_indices[:, 1]].transpose()

            hat_g_cx = (g_cx - p_cx) / p_w
            hat_g_cy = (g_cy - p_cy) / p_h
            hat_g_w = np.log(g_w / p_w)
            hat_g_h = np.log(g_h / p_h)

            hat_g = np.stack([hat_g_cx, hat_g_cy, hat_g_w, hat_g_h], axis=1)
            y_true_loc[match_indices[:, 1]] = hat_g
            y_true.append(np.concatenate([y_true_clf, y_true_loc], axis=1))

        return images, np.stack(y_true)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()


def calculate_iou(gt_boxes, pr_boxes):
    exp_gt_boxes = gt_boxes[:, None]  # Ground truth box가 행 기준으로 정렬되도록
    exp_pr_boxes = pr_boxes[None]  # prior box가 열 기준으로 정렬되도록

    # Calculate Intersection
    gt_cx, gt_cy, gt_w, gt_h = np.split(exp_gt_boxes, 4, axis=-1)
    pr_cx, pr_cy, pr_w, pr_h = np.split(exp_pr_boxes, 4, axis=-1)

    gt_xmin, gt_xmax = gt_cx - gt_w / 2, gt_cx + gt_w / 2
    gt_ymin, gt_ymax = gt_cy - gt_h / 2, gt_cy + gt_h / 2
    pr_xmin, pr_xmax = pr_cx - pr_w / 2, pr_cx + pr_w / 2
    pr_ymin, pr_ymax = pr_cy - pr_h / 2, pr_cy + pr_h / 2

    in_xmin = np.maximum(gt_xmin, pr_xmin)
    in_xmax = np.minimum(gt_xmax, pr_xmax)
    in_width = in_xmax - in_xmin
    in_width[in_width < 0] = 0

    in_ymin = np.maximum(gt_ymin, pr_ymin)
    in_ymax = np.minimum(gt_ymax, pr_ymax)
    in_height = in_ymax - in_ymin
    in_height[in_height < 0] = 0

    intersection = in_width * in_height
    intersection = np.squeeze(intersection, axis=-1)  # drop last dimension

    # Calculate Union
    gt_sizes = exp_gt_boxes[..., 2] * exp_gt_boxes[..., 3]
    pr_sizes = exp_pr_boxes[..., 2] * exp_pr_boxes[..., 3]

    union = (gt_sizes + pr_sizes) - intersection

    # Calculate Intersection Over Union
    return (intersection / (union + 1e-5))
