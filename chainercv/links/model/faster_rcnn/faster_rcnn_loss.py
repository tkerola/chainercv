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
        self.n_fg_class = faster_rcnn.n_fg_class

        # These parameters need to be consistent across modules
        proposal_target_creator_params.update({
            'n_fg_class': self.n_fg_class,
            'loc_normalize_mean': self.faster_rcnn.loc_normalize_mean,
            'loc_normalize_std': self.faster_rcnn.loc_normalize_std,
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
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('currently only batch size 1 is supported')

        img_size = imgs.shape[2:][::-1]

        features = self.faster_rcnn.extractor(imgs, test=not self.train)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale, test=not self.train)

        # since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_cls_loc, gt_roi_label, roi_loc_in_weight, \
            roi_loc_out_weight = self.proposal_target_creator(roi, bbox, label)
        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index, test=not self.train)

        # RPN losses
        gt_rpn_loc, gt_rpn_label, rpn_loc_in_weight, rpn_loc_out_weight =\
            self.anchor_target_creator(bbox, anchor, img_size)

        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)
        rpn_loc_loss = smooth_l1_loss(
            rpn_loc, gt_rpn_loc,
            rpn_loc_in_weight, rpn_loc_out_weight, self.rpn_sigma)

        # Losses for outputs of the head.
        cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)
        loc_loss = smooth_l1_loss(
            roi_cls_loc, gt_roi_cls_loc,
            roi_loc_in_weight, roi_loc_out_weight, self.sigma)
        loc_loss /= len(roi_cls_loc)

        loss = rpn_loc_loss + rpn_cls_loss + loc_loss + cls_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'loc_loss': loc_loss,
                                 'cls_loss': cls_loss,
                                 'loss': loss},
                                self)
        return loss
