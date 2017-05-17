import numpy as np

from chainer import cuda

from chainercv.links.model.faster_rcnn.utils.bbox_regression_target import \
    bbox_regression_target
from chainercv.utils.bbox.bbox_overlap import bbox_overlap


class ProposalTargetCreator(object):
    """Assign proposals to ground-truth targets.

    The :meth:`__call__` of this class generates training targets/labels
    for each object proposal.
    This is used to train Faster RCNN [1].

    .. [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_class (int): Number of classes to categorize.
        batch_size (int): Number of regions to produce.
        bbox_normalize_mean (tuple of four floats): Mean values to normalize
            coordinates of bouding boxes.
        bbox_normalize_std (tupler of four floats): Standard deviation of
            the coordinates of bounding boxes.
        bbox_inside_weight (tuple of four floats):
        fg_fraction (float): Fraction of regions that is labeled foreground.
        fg_thresh (float): Overlap threshold for a ROI to be considered
            foreground.
        bg_thresh_hi (float): ROI is considered to be background if overlap is
            in [:obj:`bg_thresh_hi`, :obj:`bg_thresh_hi`).
        bg_thresh_lo (float): See :obj:`bg_thresh_hi`.

    """

    def __init__(self, n_class=21,
                 batch_size=128,
                 bbox_normalize_mean=(0., 0., 0., 0.),
                 bbox_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 bbox_inside_weight=(1., 1., 1., 1.),
                 fg_fraction=0.25,
                 fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0
                 ):
        self.n_class = n_class
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.bbox_inside_weight = bbox_inside_weight
        self.bbox_normalize_mean = bbox_normalize_mean
        self.bbox_normalize_std = bbox_normalize_std
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo

    def __call__(self, roi, raw_bbox, label):
        """Assigns labels to sampled proposals from RPN.

        This samples total of :obj:`self.batch_size` RoIs from concatenated
        list of bounding boxes from :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels and bounding
        box offsets.
        As many as :obj:`fg_fraction * self.batch_size` RoIs are
        sampled with foreground label assignments.

        The second axis of the bounding box arrays contain coordinates
        of bounding boxes which are ordered by
        :obj:`(x_min, y_min, x_max, y_max)`.
        Offsets of bounding boxes are calculated using
        :func:`chainercv.links.bbox_regression_target`.
        Also, types of inputs and outputs are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.batch_size`.
        * :math:`K` is number of object classes.

        Args:
            roi (array): Region of interests from which we sample.
                This is an array whose shape is :math:`(R, 4)`
            raw_bbox (array): The ground truth bounding boxes. Its shape is \
                :math:`(R', 4)`.
            label (array): The ground truth bounding box labels. Its shape \
                is :math:`(R',)`.

        Returns:
            (array, array, array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **roi_bbox**: Bounding boxes that are sampled. \
                Its shape is :math:`(S, K \\times 4)`. The last \
                axis represents bounding box offsets for each of the \
                :math:`K` classes. The coordinates for the same class is \
                contiguous in this array. The coordinates are ordered by \
                :obj:`x_min, y_min, x_max, y_max`.
            * **roi_gt_label**: Labels sampled for training. Its shape is \
                :math:`(S,)`.
            * **roi_bbox_inside_weight**: Inside weights used to \
                compute losses for Faster RCNN. Its shape is \
                :math:`(S, K \\times 4)`. The second axis is organized \
                similarly to :obj:`roi_bbox_target`.
            * **roi_bbox_outside_weight**: Outside weights used to compute \
                losses for Faster RCNN. Its shape is \
                :math:`(S, K \\times 4)`. The second axis is organized \
                similarly to :obj:`roi_bbox_target`.

        """
        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        raw_bbox = cuda.to_cpu(raw_bbox)
        label = cuda.to_cpu(label)

        n_bbox, _ = raw_bbox.shape

        roi = np.concatenate((roi, raw_bbox), axis=0)

        fg_roi = np.round(self.fg_fraction * self.batch_size)

        # Sample rois with classification labels and bounding box regression
        # targets
        sample_roi, roi_gt_bbox, roi_gt_label, roi_bbox_inside_weight =\
            self._sample_roi(
                roi, raw_bbox, label, fg_roi,
                self.batch_size, self.n_class)
        roi_gt_label = roi_gt_label.astype(np.int32)
        sample_roi = sample_roi.astype(np.float32)

        roi_bbox_outside_weight =\
            (roi_bbox_inside_weight > 0).astype(np.float32)

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            roi_gt_bbox = cuda.to_gpu(roi_gt_bbox)
            roi_gt_label = cuda.to_gpu(roi_gt_label)
            roi_bbox_inside_weight = cuda.to_gpu(roi_bbox_inside_weight)
            roi_bbox_outside_weight = cuda.to_gpu(roi_bbox_outside_weight)
        return sample_roi, roi_gt_bbox, roi_gt_label,\
            roi_bbox_inside_weight, roi_bbox_outside_weight

    def _sample_roi(
            self, roi, raw_bbox, label, fg_roi_per_image, roi_per_image,
            n_class):
        # Generate a random sample of RoIs comprising foreground and background
        # examples.
        overlap = bbox_overlap(roi, raw_bbox)
        gt_assignment = overlap.argmax(axis=1)
        max_overlap = overlap.max(axis=1)
        roi_gt_label = label[gt_assignment]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_index = np.where(max_overlap >= self.fg_thresh)[0]
        # Guard against the case when an image has fewer than fg_roi_per_image
        # foreground RoIs
        fg_roi_per_this_image = int(min(fg_roi_per_image, fg_index.size))
        # Sample foreground regions without replacement
        if fg_index.size > 0:
            fg_index = np.random.choice(
                fg_index, size=fg_roi_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_index = np.where((max_overlap < self.bg_thresh_hi) &
                            (max_overlap >= self.bg_thresh_lo))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_roi_per_this_image = roi_per_image - fg_roi_per_this_image
        bg_roi_per_this_image = int(min(bg_roi_per_this_image, bg_index.size))
        # Sample background regions without replacement
        if bg_index.size > 0:
            bg_index = np.random.choice(
                bg_index, size=bg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_index = np.append(fg_index, bg_index)
        # Select sampled values from various arrays:
        roi_gt_label = roi_gt_label[keep_index]
        # Clamp labels for the background RoIs to 0
        roi_gt_label[fg_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        roi_gt_bbox = bbox_regression_target(
            sample_roi, raw_bbox[gt_assignment[keep_index]])
        # Normalize targets by a precomputed mean and stdev
        roi_gt_bbox = ((roi_gt_bbox - np.array(self.bbox_normalize_mean)
                 ) / np.array(self.bbox_normalize_std))

        roi_gt_bbox, roi_bbox_inside_weight = \
            _get_bbox_regression_label(
                roi_gt_bbox, roi_gt_label, n_class, self.bbox_inside_weight)
        return sample_roi, roi_gt_bbox, roi_gt_label, roi_bbox_inside_weight


def _get_bbox_regression_label(bbox, label, n_class, bbox_inside_weight_coeff):
    # Bounding-box regression targets (bbox_target_data) are stored in a
    # compact form S x (class, tx, ty, tw, th)
    # This function expands those targets into the 4-of-4*K representation
    # used by the network (i.e. only one class has non-zero targets).

    # Returns:
    #     bbox_target (ndarray): S x 4K blob of regression targets
    #     roi_bbox_inside_weights (ndarray): S x 4K blob of loss weights

    n_bbox = label.shape[0]
    bbox_target = np.zeros((n_bbox, 4 * n_class), dtype=np.float32)
    bbox_inside_weight = np.zeros_like(bbox_target)
    index = np.where(label > 0)[0]
    for ind in index:
        cls = int(label[ind])
        start = int(4 * cls)
        end = int(start + 4)
        bbox_target[ind, start:end] = bbox[ind]
        bbox_inside_weight[ind, start:end] = bbox_inside_weight_coeff
    return bbox_target, bbox_inside_weight
