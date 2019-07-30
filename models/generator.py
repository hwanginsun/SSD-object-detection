"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical
from utils.dataset import  DetectionDataset


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
        self.config = {
            "strides": self.strides,
            "scales": self.scales,
            "ratios": self.ratios
        }

    def generate(self, image_shape):
        """
        image_shape에 맞춰서, Prior Box(==Default Boxes)를 구성

        return :
        (# Prior boxes, 4)로 이루어진 출력값 생성
        """
        height, width = image_shape[:2]
        multi_boxes = []
        for stride, df in self.bbox_df.groupby('stride'):
            boxes = []
            for idx, row in df.iterrows():
                stride, box_width, box_height = row.stride, row.w, row.h
                ys, xs = np.mgrid[0:height:stride, 0:width:stride]
                box_width = np.ones_like(xs) * box_width
                box_height = np.ones_like(ys) * box_height
                center_xs = stride // 2 + xs
                center_ys = stride // 2 + ys

                block_centers = np.stack((center_xs, center_ys,
                                          box_width, box_height),
                                         axis=-1)
                boxes.append(block_centers)
            boxes = np.stack(boxes, axis=2)
            boxes = np.reshape(boxes, (-1, 4))
            multi_boxes.append(boxes)
        multi_boxes = np.concatenate(multi_boxes, axis=0)
        return multi_boxes

    def setup(self):
        bbox_df = pd.DataFrame(columns=['stride', 'w', 'h'])
        for scale, stride in zip(self.scales, self.strides):
            for ratio in self.ratios:
                w = np.round(scale * ratio[0]).astype(np.int)
                h = np.round(scale * ratio[1]).astype(np.int)
                bbox_df.loc[len(bbox_df) + 1] = [stride, w, h]

        bbox_df.stride = bbox_df.stride.astype(np.int)
        bbox_df.w = bbox_df.w.astype(np.int)
        bbox_df.h = bbox_df.h.astype(np.int)
        self.bbox_df = bbox_df

    def get_config(self):
        return self.config


class DetectionGenerator(Sequence):
    'Generates Localization dataset for Keras'
    def __init__(self, dataset:DetectionDataset, prior:PriorBoxes,
                 batch_size=32, best_match_policy=False, shuffle=True):
        'Initialization'
        # Dictionary로 받았을 때에만 Multiprocessing이 동작가능함.
        # Keras fit_generator에서 Multiprocessing으로 동작시키기 위함
        if isinstance(dataset, dict):
            self.dataset = DetectionDataset(**dataset)
        elif isinstance(dataset, DetectionDataset):
            self.dataset = dataset
        else:
            raise ValueError('dataset은 dict혹은 DetectionDataset Class로 이루어져 있어야 합니다.')

        if isinstance(prior, dict):
            self.prior = PriorBoxes(**prior)
        elif isinstance(prior, PriorBoxes):
            self.prior = prior
        else:
            raise ValueError('PriorBoxes은 dict 혹은 PriorBoxes Class로 이루어져 있어야 합니다.')

        self.batch_size = batch_size
        self.best_match_policy = best_match_policy
        self.shuffle = shuffle
        self.num_classes = self.dataset.num_classes
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

            gt_boxes = gt_df[['cx', 'cy', 'w', 'h']].values
            gt_labels = gt_df['label'].values
            iou = calculate_iou(gt_boxes, pr_boxes)

            if self.best_match_policy:
                best_indices = np.stack([np.arange(iou.shape[0]),
                                         np.argmax(iou, axis=1)], axis=1)
                match_indices = np.argwhere(iou >= 0.5)
                match_indices = np.concatenate([match_indices, best_indices])

            gt_match_indices = match_indices[:, 0]
            pr_match_indices = match_indices[:, 1]

            # Background로 일단 채움
            y_true_clf = np.ones((pr_boxes.shape[0])) * self.num_classes
            y_true_clf[pr_match_indices] = gt_labels[gt_match_indices]
            if self.best_match_policy:
                ignore_indices = np.argwhere((iou < 0.5) & (iou >= 0.4))[:, 1]
                y_true_clf[ignore_indices] = -1

            # classification One-Hot Encoding
            y_true_clf = to_categorical(y_true_clf,
                                        num_classes=self.num_classes + 1)

            # Positional Information Encoding
            y_true_loc = np.zeros((pr_boxes.shape[0], 4))
            g_cx, g_cy, g_w, g_h = gt_boxes[gt_match_indices].transpose()
            p_cx, p_cy, p_w, p_h = pr_boxes[pr_match_indices].transpose()

            hat_g_cx = (g_cx - p_cx) / p_w
            hat_g_cy = (g_cy - p_cy) / p_h
            hat_g_w = np.log(g_w / p_w)
            hat_g_h = np.log(g_h / p_h)

            hat_g = np.stack([hat_g_cx, hat_g_cy, hat_g_w, hat_g_h], axis=1)
            y_true_loc[pr_match_indices] = hat_g

            y_true_head = np.concatenate([y_true_clf, y_true_loc], axis=1)

            y_true.append(y_true_head)

        return images, np.stack(y_true)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()


def calculate_iou(gt_boxes,pr_boxes):
    # 1. pivot bounding boxes
    exp_gt_boxes = gt_boxes[:,None] # Ground truth box가 행 기준으로 정렬되도록
    exp_pr_boxes = pr_boxes[None,:] # prior box가 열 기준으로 정렬되도록

    # 2. calculate intersection over union
    # 2.1. Calculate Intersection
    gt_cx, gt_cy, gt_w, gt_h = exp_gt_boxes.transpose(2,0,1)
    pr_cx, pr_cy, pr_w, pr_h = exp_pr_boxes.transpose(2,0,1)

    # (cx,cy,w,h) -> (xmin,ymin,xmax,ymax)
    gt_xmin, gt_xmax = gt_cx-gt_w/2, gt_cx+gt_w/2
    gt_ymin, gt_ymax = gt_cy-gt_h/2, gt_cy+gt_h/2
    pr_xmin, pr_xmax = pr_cx-pr_w/2, pr_cx+pr_w/2
    pr_ymin, pr_ymax = pr_cy-pr_h/2, pr_cy+pr_h/2

    # 겹친 사각형의 너비와 높이 구하기
    in_xmin = np.maximum(gt_xmin, pr_xmin)
    in_xmax = np.minimum(gt_xmax, pr_xmax)
    in_width = np.maximum(0, in_xmax - in_xmin)

    in_ymin = np.maximum(gt_ymin, pr_ymin)
    in_ymax = np.minimum(gt_ymax, pr_ymax)
    in_height = np.maximum(0, in_ymax - in_ymin)

    # 겹친 사각형의 넓이 구하기
    intersection = in_width*in_height

    gt_sizes = exp_gt_boxes[...,2] * exp_gt_boxes[...,3]
    pr_sizes = exp_pr_boxes[...,2] * exp_pr_boxes[...,3]

    # 2.2. Calculate Union
    union = (gt_sizes + pr_sizes) - intersection

    # 0 나누기 방지를 위함
    return (intersection / (union+1e-5))


def restore_position(predict_boxes, pr_boxes):
    res_cx = (predict_boxes[:, 0]
              * pr_boxes[:, 2]
              + pr_boxes[:, 0])
    res_cy = (predict_boxes[:, 1]
              * pr_boxes[:, 3]
              + pr_boxes[:, 1])
    res_w = (np.exp(predict_boxes[:, 2])
             * pr_boxes[:, 2])
    res_h = (np.exp(predict_boxes[:, 3])
             * pr_boxes[:, 3])
    return np.stack([res_cx, res_cy, res_w, res_h], axis=-1)
