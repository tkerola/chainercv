from __future__ import division

import os
import six
import warnings

import chainer
from chainer.dataset.download import get_dataset_directory
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.model.ssd import Multibox
from chainercv.links.model.ssd import Normalize
from chainercv.links.model.ssd import SSD
from chainercv.utils import download

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


_imagenet_mean = (104, 117, 123)


class VGG16(chainer.Chain):
    """An extended VGG-16 model for SSD300 and SSD512.

    This is an extended VGG-16 model proposed in [#]_.
    The differences from original VGG-16 [#]_ are shown below.

    * :obj:`conv5_1`, :obj:`conv5_2` and :obj:`conv5_3` are changed from \
    :class:`~chainer.links.Convolution2d` to \
    :class:`~chainer.links.DilatedConvolution2d`.
    * :class:`~chainercv.links.model.ssd.Normalize` is \
    inserted after :obj:`conv4_3`.
    * The parameters of max pooling after :obj:`conv5_3` are changed.
    * :obj:`fc6` and :obj:`fc7` are converted to :obj:`conv6` and :obj:`conv7`.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    .. [#] Karen Simonyan, Andrew Zisserman.
       Very Deep Convolutional Networks for Large-Scale Image Recognition.
       ICLR 2015.
    """

    def __init__(self, **links):
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(None, 64, 3, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, pad=1),

            conv2_1=L.Convolution2D(None, 128, 3, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, pad=1),

            conv3_1=L.Convolution2D(None, 256, 3, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, pad=1),

            conv4_1=L.Convolution2D(None, 512, 3, pad=1),
            conv4_2=L.Convolution2D(None, 512, 3, pad=1),
            conv4_3=L.Convolution2D(None, 512, 3, pad=1),
            norm4=Normalize(512, initial=initializers.Constant(20)),

            conv5_1=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_2=L.DilatedConvolution2D(None, 512, 3, pad=1),
            conv5_3=L.DilatedConvolution2D(None, 512, 3, pad=1),

            conv6=L.DilatedConvolution2D(None, 1024, 3, pad=6, dilate=6),
            conv7=L.Convolution2D(None, 1024, 1),
        )
        for name, link in six.iteritems(links):
            self.add_link(name, link)

    def __call__(self, x):
        ys = list()

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        ys.append(self.norm4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        ys.append(h)

        return ys


class VGG16Extractor300(VGG16):
    """A VGG-16 based feature extractor for SSD300.

    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD300`.
    This extractor is based on :class:`~chainercv.links.model.ssd.VGG16`.
    """

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(VGG16Extractor300, self).__init__(
            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, **init),
        )

    def __call__(self, x):
        """Compute feature maps from a batch of images.

        This method extracts feature maps from
        :obj:`conv4_3`, :obj:`conv7`, :obj:`conv8_2`,
        :obj:`conv9_2`, :obj:`conv10_2`, and :obj:`conv11_2`.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`300\\times 300`.

        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = super(VGG16Extractor300, self).__call__(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


class VGG16Extractor512(VGG16):
    """A VGG-16 based feature extractor for SSD512.

    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD512`.
    This extractor is based on :class:`~chainercv.links.model.ssd.VGG16`.
    """

    insize = 512
    grids = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(VGG16Extractor512, self).__init__(
            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv12_1=L.Convolution2D(None, 128, 1, **init),
            conv12_2=L.Convolution2D(None, 256, 4,  pad=1, **init),
        )

    def __call__(self, x):
        """Compute feature maps from a batch of images.

        This method extracts feature maps from
        :obj:`conv4_3`, :obj:`conv7`, :obj:`conv8_2`,
        :obj:`conv9_2`, :obj:`conv10_2`, :obj:`conv11_2`, and :obj:`conv12_2`.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`512\\times 512`.

        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = super(VGG16Extractor512, self).__call__(x)
        for i in range(8, 12 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)
        return ys


def _check_pretrained_model(n_fg_class, pretrained_model, models):
    if pretrained_model in models:
        model = models[pretrained_model]
        if n_fg_class and not n_fg_class == model['n_fg_class']:
            raise ValueError('n_fg_class mismatch')
        n_fg_class = model['n_fg_class']

        root = get_dataset_directory('pfnet/chainercv/models')
        basename = os.path.basename(model['url'])
        path = os.path.join(root, basename)
        if not os.path.exists(path):
            download_file = download.cached_download(model['url'])
            os.rename(download_file, path)

        if not _available:
            warnings.warn(
                'cv2 is not installed on your environment. '
                'Pretrained models are trained with cv2. '
                'The performace may change with Pillow backend.',
                RuntimeWarning)
    elif pretrained_model:
        path = pretrained_model
    else:
        path = None

    return n_fg_class, path


class SSD300(SSD):
    """Single Shot Multibox Detector with 300x300 inputs.

    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`~chainercv.links.model.ssd.VGG16Extractor300` as
    its feature extractor.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the Caffe model provided by \
                `the original implementation \
                <https://github.com/weiliu89/caffe/tree/ssd>`_. \
                The conversion code is `chainercv/examples/ssd/caffe2npz.py`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _models = {
        'voc0712': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.1/ssd300_voc0712.npz'
        }
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        n_fg_class, path = _check_pretrained_model(
            n_fg_class, pretrained_model, self._models)

        super(SSD300, self).__init__(
            extractor=VGG16Extractor300(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=[s / 300 for s in (8, 16, 32, 64, 100, 300)],
            sizes=[s / 300 for s in (30, 60, 111, 162, 213, 264, 315)],
            mean=_imagenet_mean)

        if path:
            chainer.serializers.load_npz(path, self)


class SSD512(SSD):
    """Single Shot Multibox Detector with 512x512 inputs.

    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`~chainercv.links.model.ssd.VGG16Extractor512` as
    its feature extractor.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
           The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the Caffe model provided by \
                `the original implementation \
                <https://github.com/weiliu89/caffe/tree/ssd>`_. \
                The conversion code is `chainercv/examples/ssd/caffe2npz.py`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _models = {
        'voc0712': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.1/ssd512_voc0712.npz'
        }
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        n_fg_class, path = _check_pretrained_model(
            n_fg_class, pretrained_model, self._models)

        super(SSD512, self).__init__(
            extractor=VGG16Extractor512(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=(
                    (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=[s / 512 for s in (8, 16, 32, 64, 128, 256, 512)],
            sizes=[s / 512 for s in
                   (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)],
            mean=_imagenet_mean)

        if path:
            chainer.serializers.load_npz(path, self)
