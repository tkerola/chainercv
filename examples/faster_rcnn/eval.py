import argparse
import matplotlib.pyplot as plot
import numpy as np

from chainer import cuda
from chainer import serializers

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainercv.evaluations import eval_detection_voc

from chainercv.datasets import VOCDetectionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu')
    parser.add_argument('model')
    args = parser.parse_args()

    model = FasterRCNNVGG16(n_class=21, score_thresh=0.05)
    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        cuda.get_device(args.gpu).use()

    dataset = VOCDetectionDataset(
        mode='test', year='2007', use_difficult=True, return_difficult=True)

    gt_bboxes = []
    gt_labels = []
    gt_difficults = []
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    for i in range(len(dataset)):
        img, bbox, label, difficult = dataset[i]
        gt_bboxes.append(bbox)
        gt_labels.append(label)
        gt_difficults.append(difficult)

        out = model.predict(img[np.newaxis])
        pred_bboxes.append(out[0][0])
        pred_labels.append(out[1][0])
        pred_scores.append(out[2][0])

    metric = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes,
        gt_labels, n_class=len(voc_utils.pascal_voc_labels),
        gt_difficults=gt_difficults,
        minoverlap=0.5, use_07_metric=True)
    print metric


if __name__ == '__main__':
    main()
