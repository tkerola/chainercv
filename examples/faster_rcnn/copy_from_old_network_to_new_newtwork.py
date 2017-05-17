
import numpy as np
import chainer
from chainercv.links import FasterRCNNVGG16


npz = np.load('result/model2017-05-10_03:16:56')
model = FasterRCNNVGG16(n_class=21)

not_used_keys = []
for key, val in npz.items():
    splitted = key.split('/')
    if splitted[0] == 'feature':
        print('copying {}'.format(key))
        first = getattr(model, splitted[0])
        second = getattr(first, splitted[1])
        third = getattr(second, splitted[2])
        third.data[:] = val
    else:
        not_used_keys.append(key)

model.rpn.conv1.W.data[:] = npz['rpn/rpn_conv_3x3/W']
model.rpn.conv1.b.data[:] = npz['rpn/rpn_conv_3x3/b']
model.rpn.score.W.data[:] = npz['rpn/rpn_cls_score/W']
model.rpn.score.b.data[:] = npz['rpn/rpn_cls_score/b']
model.rpn.bbox.W.data[:] = npz['rpn/rpn_bbox_pred/W']
model.rpn.bbox.b.data[:] = npz['rpn/rpn_bbox_pred/b']
model.head.fc6.W.data[:] = npz['head/fc6/W']
model.head.fc6.b.data[:] = npz['head/fc6/b']
model.head.fc7.W.data[:] = npz['head/fc7/W']
model.head.fc7.b.data[:] = npz['head/fc7/b']
model.head.bbox.W.data[:] = npz['head/bbox_pred/W']
model.head.bbox.b.data[:] = npz['head/bbox_pred/b']
model.head.score.W.data[:] = npz['head/cls_score/W']
model.head.score.b.data[:] = npz['head/cls_score/b']


chainer.serializers.save_npz('working_weight_new', model)

