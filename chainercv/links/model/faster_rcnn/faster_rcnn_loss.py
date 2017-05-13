import chainer
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
        self.roi_size = faster_rcnn.roi_size
        self.spatial_scale = faster_rcnn.spatial_scale

        # These parameters need to be consistent across modules
        proposal_target_creator_params.update({
            'n_class': self.n_class,
            'bbox_normalize_target_precomputed':
            self.faster_rcnn.target_precomputed,
            'bbox_normalize_mean': self.faster_rcnn.bbox_normalize_mean,
            'bbox_normalize_std': self.faster_rcnn.bbox_normalize_std,
        })
        self.proposal_target_creator = ProposalTargetCreator(
            **proposal_target_creator_params)
        self.anchor_target_creator = AnchorTargetCreator(
            **anchor_target_creator_params)

        self.train = True

    def __call__(self, img, bbox, label, scale=1.):
        if isinstance(bbox, chainer.Variable):
            bbox = bbox.data
        if isinstance(label, chainer.Variable):
            label = label.data

        img_size = img.shape[2:][::-1]
        layers = ['feature', 'rpn_bbox_pred', 'rpn_cls_score',
                  'roi', 'anchor']
        out = self.faster_rcnn(
            img, scale=scale, layers=layers, rpn_only=True,
            test=not self.train)

        # RPN losses
        n, _, hh, ww = out['rpn_bbox_pred'].shape
        rpn_bbox_target, rpn_label, rpn_bbox_inside_weight, \
            rpn_bbox_outside_weight = self.anchor_target_creator(
                bbox, out['anchor'], (ww, hh), img_size)
        rpn_label = rpn_label.reshape((n, -1))
        rpn_cls_score = out['rpn_cls_score'].reshape(1, 2, -1)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_cls_score, rpn_label)
        rpn_loss_bbox = smooth_l1_loss(
            out['rpn_bbox_pred'],
            rpn_bbox_target,
            rpn_bbox_inside_weight,
            rpn_bbox_outside_weight, self.rpn_sigma)

        # Sample RoIs and forward
        roi_sample, bbox_target_sample, label_sample, bbox_inside_weight, \
            bbox_outside_weight = self.proposal_target_creator(
                out['roi'], bbox, label)
        pool5 = F.roi_pooling_2d(
            out['feature'],
            roi_sample, self.roi_size, self.roi_size, self.spatial_scale)
        out = self.faster_rcnn.head(pool5, train=self.train)
        bbox_tf = out['bbox_tf']
        score = out['score']

        # Losses for outputs of the head.
        loss_cls = F.softmax_cross_entropy(score, label_sample)
        loss_bbox = smooth_l1_loss(
            bbox_tf, bbox_target_sample,
            bbox_inside_weight, bbox_outside_weight, self.sigma)

        loss = rpn_loss_bbox + rpn_cls_loss + loss_bbox + loss_cls
        chainer.reporter.report({'rpn_loss_cls': rpn_cls_loss,
                                 'rpn_loss_bbox': rpn_loss_bbox,
                                 'loss_bbox': loss_bbox,
                                 'loss_cls': loss_cls,
                                 'loss': loss},
                                self)
        return loss
