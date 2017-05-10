from __future__ import division

import numpy as np
import six


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and fit to given aspect ratios.
    Area of a scaled anchor is preserved when fitting to an aspect ratio.

    There will be :obj:`R = len(ratios) * len(scales)` anchors generated.
    The :obj:`i * len(scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`scales[j]`.

    For example, if scale is :math:`8` and ratio is :math:`0.25`,
    width and height of the base window will be stretched by :math:`8` for
    scaling. For fitting the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): Width and height of the reference window.
        ratios (list of floats): Anchors with ratios contained in this list
            will be generated. Ratio is the ratio of the height by the width.
        scales (list of numbers): Values in :obj:`scales` determine area of
            possibly generated anchors. Those areas will be square of an
            element in :obj:`scales` times the original area of the
            reference window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`. The second axis contains four \
        values :obj:`(x_min, y_min, x_max, y_max)`, \
        which are coordinates of the bottom left and the top right vertices.

    """
    px = base_size / 2.
    py = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(scales)):
            w = base_size * scales[j] * np.sqrt(1. / ratios[i])
            h = base_size * scales[j] * np.sqrt(ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = px - w / 2.
            anchor_base[index, 1] = py - h / 2.
            anchor_base[index, 2] = px + w / 2.
            anchor_base[index, 3] = py + h / 2.
    return anchor_base