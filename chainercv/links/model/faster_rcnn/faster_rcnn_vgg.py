import collections
import numpy as np
import os

import chainer
from chainer.dataset.download import get_dataset_directory
import chainer.functions as F
import chainer.links as L
from chainer.links import VGG16Layers

from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNNBase
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.utils import download


urls = {
    'voc07': 'https://github.com/yuyu2172/share-weights/releases/'
    'download/0.0.1/faster_rcnn_vgg_voc07.npz'
}
n_fg_classes = {'voc07': 20}


def _relu(x):
    # use_cudnn = False is sometimes x3 faster than otherwise.
    # This will be the default mode in Chainer v2.
    return F.relu(x, use_cudnn=False)


class FasterRCNNVGG16(FasterRCNNBase):

    """Faster R-CNN based on VGG16.

    When you specify the path of the pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows.

    * :obj:`voc07`: Loads weights trained with trainval split of \
        PASCAL VOC2007 Detection Dataset.
    * :obj:`imagenet`: Loads weights trained with ImageNet Classfication \
        task for the feature extractor and the head modules. \
        Weights that do not have a corresponding layer in VGG16 network \
        will be randomly initialized.

    For descriptions on the interface of this model, please refer to
    :class:`chainercv.links.model.faster_rcnn.FasterRCNNBase`.

    :obj:`FasterRCNNVGG16` supports finer control on random initialization of
    weights by arguments
    :obj:`vgg_initialW`, :obj:`rpn_initialW`, :obj:`loc_initialW` and
    :obj:`score_initialW`.
    It accepts a callable that takes an array and edits its values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this argument is specified as in one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        nms_thresh (float): Threshold value used when calling NMS in
            :func:`predict`.
        score_thresh (float): Threshold value used to discard low
            confidence proposals in :func:`predict`.
        min_size (int): A preprocessing paramter for :func:`prepare`.
        max_size (int): A preprocessing paramter for :func:`prepare`.
        ratios (list of floats): Anchors with ratios contained in this list
            will be generated. Ratio is the ratio of the height by the width.
        anchor_scales (list of numbers): Values in :obj:`anchor_scales`
            determine area of possibly generated anchors. Those areas will be
            square of an element in :obj:`anchor_scales` times the original
            area of the reference window.
        vgg_initialW (callable): Initializer for the layers corresponding to
            VGG16 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :obj:`chainercv.links.model.faster_rcnn.ProposalCreator`.

    """

    feat_stride = 16

    def __init__(self,
                 n_fg_class=None,
                 pretrained_model='voc07',
                 nms_thresh=0.3, score_thresh=0.7,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 vgg_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params={}
                 ):
        if n_fg_class is None:
            if pretrained_model not in n_fg_classes:
                raise ValueError(
                    'The n_fg_class needs to be supplied as an argument')
            n_fg_class = n_fg_classes[pretrained_model]

        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if vgg_initialW is None and pretrained_model:
            vgg_initialW = chainer.initializers.constant.Zero()

        extractor = VGG16FeatureExtractor(initialW=vgg_initialW)
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = VGG16RoIPoolingHead(
            n_fg_class,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            vgg_initialW=vgg_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
            n_fg_class=n_fg_class,
            mean=np.array([102.9801, 115.9465, 122.7717],
                          dtype=np.float32)[:, None, None],
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model in urls:
            data_root = get_dataset_directory('pfnet/chainercv/models')
            url = urls[pretrained_model]
            fn = url.rsplit('/', 1)[-1]
            dest_fn = os.path.join(data_root, fn)
            if not os.path.exists(dest_fn):
                download_file = download.cached_download(url)
                os.rename(download_file, dest_fn)
            chainer.serializers.load_npz(dest_fn, self)
        elif pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_vgg16()
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _copy_imagenet_pretrained_vgg16(self):
        pretrained_model = VGG16Layers()
        self.extractor.conv1_1.copyparams(pretrained_model.conv1_1)
        self.extractor.conv1_2.copyparams(pretrained_model.conv1_2)
        self.extractor.conv2_1.copyparams(pretrained_model.conv2_1)
        self.extractor.conv2_2.copyparams(pretrained_model.conv2_2)
        self.extractor.conv3_1.copyparams(pretrained_model.conv3_1)
        self.extractor.conv3_2.copyparams(pretrained_model.conv3_2)
        self.extractor.conv3_3.copyparams(pretrained_model.conv3_3)
        self.extractor.conv4_1.copyparams(pretrained_model.conv4_1)
        self.extractor.conv4_2.copyparams(pretrained_model.conv4_2)
        self.extractor.conv4_3.copyparams(pretrained_model.conv4_3)
        self.extractor.conv5_1.copyparams(pretrained_model.conv5_1)
        self.extractor.conv5_2.copyparams(pretrained_model.conv5_2)
        self.extractor.conv5_3.copyparams(pretrained_model.conv5_3)
        self.head.fc6.copyparams(pretrained_model.fc6)
        self.head.fc7.copyparams(pretrained_model.fc7)


class VGG16RoIPoolingHead(chainer.Chain):

    """Faster R-CNN Head for VGG16 based implementation.

    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on RoI
    features.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        roi_size (int): Height and width of the features after RoI-pooled.
        spatial_scale (float): Scale of the roi is resized.

    """

    def __init__(self, n_fg_class, roi_size, spatial_scale,
                 vgg_initialW=None, loc_initialW=None, score_initialW=None):
        # n_class includes the background
        super(VGG16RoIPoolingHead, self).__init__(
            fc6=L.Linear(25088, 4096, initialW=vgg_initialW),
            fc7=L.Linear(4096, 4096, initialW=vgg_initialW),
            cls_loc=L.Linear(4096, (n_fg_class + 1) * 4,
                             initialW=loc_initialW),
            score=L.Linear(4096, n_fg_class + 1, initialW=score_initialW)
        )
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices, test=True):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (~chainer.Variable): 4D image variable.
            rois (array): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted
                bounding boxes from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (array): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
            test (bool): Whether in test mode or not. This has no effect in
                the current implementation.

        """
        roi_indices = roi_indices.astype(np.float32)
        rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = F.roi_pooling_2d(
            x, rois, self.roi_size, self.roi_size, self.spatial_scale)

        fc6 = _relu(self.fc6(pool))
        fc7 = _relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


class VGG16FeatureExtractor(chainer.Chain):
    """Truncated VGG16 that extracts a conv5_3 feature.

    :obj:`initialW` accepts a callable that takes an array and edits its
    values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        initialW (callable): Initializer for weights.

    """

    def __init__(self, initialW=None):
        super(VGG16FeatureExtractor, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, 1, 1, initialW=initialW),
            conv1_2=L.Convolution2D(64, 64, 3, 1, 1, initialW=initialW),
            conv2_1=L.Convolution2D(64, 128, 3, 1, 1, initialW=initialW),
            conv2_2=L.Convolution2D(128, 128, 3, 1, 1, initialW=initialW),
            conv3_1=L.Convolution2D(128, 256, 3, 1, 1, initialW=initialW),
            conv3_2=L.Convolution2D(256, 256, 3, 1, 1, initialW=initialW),
            conv3_3=L.Convolution2D(256, 256, 3, 1, 1, initialW=initialW),
            conv4_1=L.Convolution2D(256, 512, 3, 1, 1, initialW=initialW),
            conv4_2=L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW),
            conv4_3=L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW),
            conv5_1=L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW),
            conv5_2=L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW),
            conv5_3=L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW),
        )
        self.functions = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, _relu]),
            ('conv1_2', [self.conv1_2, _relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, _relu]),
            ('conv2_2', [self.conv2_2, _relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, _relu]),
            ('conv3_2', [self.conv3_2, _relu]),
            ('conv3_3', [self.conv3_3, _relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, _relu]),
            ('conv4_2', [self.conv4_2, _relu]),
            ('conv4_3', [self.conv4_3, _relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, _relu]),
            ('conv5_2', [self.conv5_2, _relu]),
            ('conv5_3', [self.conv5_3, _relu]),
        ])

    def __call__(self, x, test=True):
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
        return h


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
