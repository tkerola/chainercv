import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainercv.functions.smooth_l1_loss import smooth_l1_loss
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import \
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.utils.proposal_target_creator import \
    ProposalTargetCreator


class FasterRCNNLoss(chainer.Chain):

    def __init__(self, faster_rcnn, rpn_sigma=3., sigma=1.,
                 anchor_target_creator_params={},
                 proposal_target_creator_params={},
                 ):
        super(FasterRCNNLoss, self).__init__(faster_rcnn=faster_rcnn)
        self.rpn_sigma = rpn_sigma
        self.sigma = sigma
        self.n_class = faster_rcnn.n_class

        # These parameters need to be consistent across modules
        proposal_target_creator_params.update({
            'n_class': self.n_class,
            'bbox_normalize_mean': self.faster_rcnn.bbox_normalize_mean,
            'bbox_normalize_std': self.faster_rcnn.bbox_normalize_std,
        })
        self.proposal_target_creator = ProposalTargetCreator(
            **proposal_target_creator_params)
        self.anchor_target_creator = AnchorTargetCreator(
            **anchor_target_creator_params)

        self.train = True

    def __call__(self, imgs, bboxes, labels, scale):
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = np.asscalar(cuda.to_cpu(scale))
        if bboxes.shape[0] != 1:
            raise ValueError('currently only batch size 1 is supported')
        bbox = bboxes[0]
        label = labels[0]
        img_size = imgs.shape[2:][::-1]

        out = self.faster_rcnn(
            imgs, scale=scale,
            layers=['features', 'rpn_bboxes', 'rpn_scores',
                    'rois', 'batch_indices', 'anchor'],
            test=not self.train)

        # RPN losses

        n, _, hh, ww = out['rpn_bboxes'].shape
        # THIS IS SINGULAR, but ROI_* are not SINGULAR
        rpn_bbox_target, rpn_label, rpn_bbox_inside_weight, \
            rpn_bbox_outside_weight = self.anchor_target_creator(
                bbox, out['anchor'], (ww, hh), img_size)
        rpn_labels = rpn_label.reshape((n, -1))
        rpn_scores = out['rpn_scores'].reshape(1, 2, -1)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_scores, rpn_labels)
        rpn_bbox_loss = smooth_l1_loss(
            out['rpn_bboxes'],
            rpn_bbox_target[None],
            rpn_bbox_inside_weight[None],
            rpn_bbox_outside_weight[None],
            self.rpn_sigma)

        # Sample RoIs and forward
        sample_rois, roi_bbox_targets, roi_labels, roi_bbox_inside_weights, \
            roi_bbox_outside_weights = self.proposal_target_creator(
                out['rois'], bbox, label)
        roi_bboxes, roi_scores = self.faster_rcnn.head(
            out['features'], sample_rois, out['batch_indices'])

        # Losses for outputs of the head.
        cls_loss = F.softmax_cross_entropy(roi_scores, roi_labels)
        bbox_loss = smooth_l1_loss(
            roi_bboxes, roi_bbox_targets,
            roi_bbox_inside_weights, roi_bbox_outside_weights,
            self.sigma)

        loss = rpn_bbox_loss + rpn_cls_loss + bbox_loss + cls_loss
        chainer.reporter.report({'rpn_bbox_loss': rpn_bbox_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'bbox_loss': bbox_loss,
                                 'cls_loss': cls_loss,
                                 'loss': loss},
                                self)
        return loss
