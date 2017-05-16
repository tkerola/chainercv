import matplotlib
matplotlib.use('agg')
import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters

from chainercv.datasets import TransformDataset
from chainercv.datasets import VOCDetectionDataset
from chainercv import transforms

from chainercv.links import FasterRCNNVGG16
from chainercv.links import FasterRCNNLoss

from chainercv.datasets.pascal_voc.voc_utils import pascal_voc_labels

from chainer.training.triggers.manual_schedule_trigger import ManualScheduleTrigger

from merge_dataset import MergeDataset


mean_pixel = np.array([102.9801, 115.9465, 122.7717])[:, None, None]


def get_train_iter(device, train_data, batchsize=1, loaderjob=None):
    if len(device) > 1:
        train_iter = [
            chainer.iterators.MultiprocessIterator(
                i, batchsize, n_processes=loaderjob, shared_mem=10000000)
            for i in chainer.datasets.split_dataset_n_random(train_data, len(device))]
    else:
        train_iter = chainer.iterators.MultiprocessIterator(
            train_data, batch_size=batchsize, n_processes=loaderjob, shared_mem=100000000)
    return train_iter


def get_updater(train_iter, optimizer, device):
    if len(device) > 1:
        updater = updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=device)
    else:
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=device[0])
    return updater


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV example: Faster RCNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--dataset', '-d', type=str, default='voc07')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    gpu = args.gpu
    lr = args.lr
    out = args.out
    seed = args.seed
    step_size = args.step_size
    iteration = args.iteration

    print('iteration ', iteration)
    print('dataset', args.dataset)

    np.random.seed(seed)

    if args.dataset == 'voc07':
        # size 5011
        labels = pascal_voc_labels
        train_data = VOCDetectionDataset(mode='trainval', year='2007')
    elif args.dataset == 'voc07+voc12':
        labels = pascal_voc_labels
        voc07 = VOCDetectionDataset(mode='trainval', year='2007')
        voc12 = VOCDetectionDataset(mode='trainval', year='2012')
        train_data = MergeDataset([voc07, voc12])

    def get_transform(use_random):
        min_size = 600
        max_size = 1000
        def transform(in_data):
            has_difficult = len(in_data) == 4
            img, bbox, label = in_data[:3]
            img -= mean_pixel 
            # Resize bounding box to a shape
            # with the smaller edge at least at length 600
            _, H, W = img.shape
            scale = 1.
            if min(H, W) < min_size:
                scale = min_size / min(H, W)

            if scale * max(H, W) > max_size:
                scale = max(H, W) * scale / max_size

            o_W, o_H = int(W * scale), int(H * scale)
            img = transforms.resize(img, (o_W, o_H))
            bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))

            # horizontally flip
            if use_random:
                img, params = transforms.random_flip(img, x_random=True, return_param=True)
                bbox = transforms.flip_bbox(bbox, (o_W, o_H), params['x_flip'])

            if has_difficult:
                return img, bbox, label, scale, in_data[-1]
            return img, bbox, label, scale
        return transform

    train_data = TransformDataset(train_data, get_transform(True))

    model = FasterRCNNLoss(
        FasterRCNNVGG16(n_class=len(labels), score_thresh=0.05)
    )

    if gpu >= 0:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    trainer.extend(extensions.snapshot_object(
        model.faster_rcnn, 'model'), trigger=(iteration, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(step_size, 'iteration'))

    log_interval = 20, 'iteration'
    val_interval = iteration, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval
    )
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/loss_bbox',
         'main/loss_cls',
         'main/rpn_loss_cls',
         'main/rpn_loss_bbox',
         'map'
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_cls'],
                file_name='rpn_loss_cls.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/rpn_loss_bbox'],
                file_name='rpn_loss_bbox.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_cls'],
                file_name='loss_cls.png'
            ),
            trigger=plot_interval
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/loss_bbox'],
                file_name='loss_bbox.png'
            ),
            trigger=plot_interval
        )

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
