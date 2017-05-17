import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox_regression_target import \
    bbox_regression_target_inv
from chainercv.utils.bbox.non_maximum_suppression import \
    non_maximum_suppression


class ProposalCreator(object):
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    This class is used for Region Proposal Networks introduced in
    Faster RCNN [1].

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.


    Args:
        use_gpu_nms (bool): Whether to use GPU powered non maximum
            suppression (NMS) or not when possible. Default value is
            :obj:`True`.
        min_size (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self, use_gpu_nms=True,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.use_gpu_nms = use_gpu_nms
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, bbox_d, score,
                 anchor, img_size, scale=1., train=False):
        """Generate deterministic proposal regions.

        The values contained in bounding box delta array :obj:`bbox_d` are
        encoded using :func:`chainercv.links.bbox_regression_target`.

        Type of the output is same as the inputs.

        On notations, :math:`A` is the total number of anchors. This is equal
        to product of height and width of an image and number of
        anchor bases per pixel.

        .. seealso::
            :func:`~chainercv.links.bbox_regression_target`

        Args:
            bbox_d (array): Predicted regression targets for anchors.
                Its shape is :math:`(A, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(A,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(A, 4)`.
            img_size (tuple of ints): A tuple :obj:`width, height`,
                which contains image size after scaling if any.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.
            train (bool): If this is in train mode or not.
                Default value is :obj:`False`.

        Returns:
            array:
            A bounding box array containing coordinates of \
            proposal boxes. Its shape is :math:`(R, 4)`.

        """
        n_pre_nms = self.n_train_pre_nms if train else self.n_test_pre_nms
        n_post_nms = self.n_train_post_nms if train else self.n_test_post_nms

        xp = cuda.get_array_module(bbox_d)
        bbox_d = cuda.to_cpu(bbox_d)
        score = cuda.to_cpu(score)
        anchor = cuda.to_cpu(anchor)

        # Convert anchors into proposal via bbox transformations
        bbox = bbox_regression_target_inv(anchor, bbox_d)

        # Clip predicted boxes to image
        bbox[:, slice(0, 4, 2)] = np.clip(
            bbox[:, slice(0, 4, 2)], 0, img_size[0])
        bbox[:, slice(1, 4, 2)] = np.clip(
            bbox[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold
        min_size = self.min_size * scale
        ws = bbox[:, 2] - bbox[:, 0]
        hs = bbox[:, 3] - bbox[:, 1]
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        proposal = bbox[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest
        # Take top pre_nms_topN (e.g. 6000)
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        bbox = bbox[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7)
        # Take after_nms_topN (e.g. 300)
        if self.use_gpu_nms and cuda.available:
            keep = non_maximum_suppression(
                cuda.to_gpu(bbox),
                thresh=self.nms_thresh,
                score=cuda.to_gpu(score))
            keep = cuda.to_cpu(keep)
        else:
            keep = non_maximum_suppression(
                proposal,
                thresh=self.nms_thresh,
                score=score)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        bbox = bbox[keep]

        if xp != np:
            bbox = cuda.to_gpu(bbox)
        return bbox
