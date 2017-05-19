import numpy as np

import chainer
from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_overlap import bbox_overlap


class AnchorTargetCreator(object):

    """Assign anchors to ground-truth targets.

    Assigns anchors to ground-truth targets to train Region Proposal Networks
    introduced in Faster RCNN [1].

    Bounding regression targets are computed using encoding scheme
    found in :obj:`chainercv.links.bbox2loc`.

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): Number of regions to produce.
        negative_overlap (float): Anchors with overlap below this
            threshold will be assigned as negative.
        positive_overlap (float): Anchors with overlap above this
            threshold will be assigned as positive.
        fg_fraction (float): Fraction of positive regions in the
            set of all regions produced.
        bbox_in_weight (tuple of four floats): Four coefficients
            used to calculate bbox_in_weight.

    """

    def __init__(self,
                 n_sample=256,
                 negative_overlap=0.3, positive_overlap=0.7,
                 fg_fraction=0.5,
                 bbox_in_weight=(1., 1., 1., 1.)):
        self.n_sample = n_sample
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap
        self.fg_fraction = fg_fraction
        self.bbox_in_weight = bbox_in_weight

    def __call__(self, bbox, anchor, img_size):
        """Calculate targets of classification labels and bbox regressions.

        Here are notations.

        * :math:`S` is number of anchors.
        * :math:`R` is number of bounding boxes.

        Args:
            bbox (array): Coodinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`W, H`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array, array, array):

            Tuple of four arrays which contains the following elements.

            * **loc**: Bounding boxes encoded into regression \
                targets. This is an array of shape :math:`(S, 4)`.
            * **label**: Labels of bounding boxes with values \
                :obj:`(1=foreground, 0=background, -1=ignore)`. Its shape \
                is :math:`(S,)`.
            * **bbox_in_weight**: Inside weight used to compute losses \
                for Faster RCNN. Its shape is :math:`(S, 4)`.
            * **bbox_out_weight** Outside weight used to compute losses \
                for Faster RCNN. Its shape is :math:`(S, 4)`.

        """
        xp = cuda.get_array_module(bbox)
        bbox = cuda.to_cpu(bbox)
        anchor = cuda.to_cpu(anchor)

        img_W, img_H = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_W, img_H)
        anchor = anchor[inside_index]
        argmax_overlaps, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_overlaps])

        # calculate inside and outside weights weights
        bbox_in_weight = np.zeros((len(inside_index), 4), dtype=np.float32)
        bbox_in_weight[label == 1, :] = np.array(
            self.bbox_in_weight)
        bbox_out_weight = self._calc_outside_weights(inside_index, label)

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        bbox_in_weight = _unmap(
            bbox_in_weight, n_anchor, inside_index, fill=0)
        bbox_out_weight = _unmap(
            bbox_out_weight, n_anchor, inside_index, fill=0)

        if xp != np:
            loc = chainer.cuda.to_gpu(loc)
            label = chainer.cuda.to_gpu(label)
            bbox_in_weight = chainer.cuda.to_gpu(bbox_in_weight)
            bbox_out_weight = chainer.cuda.to_gpu(bbox_out_weight)
        return loc, label, bbox_in_weight, bbox_out_weight

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index), ), dtype=np.int32)
        label.fill(-1)

        argmax_overlaps, max_overlaps, gt_argmax_overlaps = \
            self._calc_overlaps(anchor, bbox, inside_index)

        # assign bg labels first so that positive labels can clobber them
        label[max_overlaps < self.negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        label[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        label[max_overlaps >= self.positive_overlap] = 1

        # subsample positive labels if we have too many
        num_fg = int(self.fg_fraction * self.n_sample)
        fg_index = np.where(label == 1)[0]
        if len(fg_index) > num_fg:
            disable_index = np.random.choice(
                fg_index, size=(len(fg_index) - num_fg), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        num_bg = self.n_sample - np.sum(label == 1)
        bg_index = np.where(label == 0)[0]
        if len(bg_index) > num_bg:
            disable_index = np.random.choice(
                bg_index, size=(len(bg_index) - num_bg), replace=False)
            label[disable_index] = -1

        return argmax_overlaps, label

    def _calc_overlaps(self, anchor, bbox, inside_index):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlap(anchor, bbox)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inside_index)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps, max_overlaps, gt_argmax_overlaps

    def _calc_outside_weights(self, inside_index, label):
        bbox_out_weight = np.zeros(
            (len(inside_index), 4), dtype=np.float32)
        # uniform weighting of examples (given non-uniform sampling)
        n_example = np.sum(label >= 0)

        positive_weight = np.ones((1, 4)) * 1.0 / n_example
        negative_weight = np.ones((1, 4)) * 1.0 / n_example

        bbox_out_weight[label == 1, :] = positive_weight
        bbox_out_weight[label == 0, :] = negative_weight

        return bbox_out_weight


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, W, H):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    xp = cuda.get_array_module(anchor)

    index_inside = xp.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= W) &  # width
        (anchor[:, 3] <= H)  # height
    )[0]
    return index_inside
