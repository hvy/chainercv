import argparse
import copy
from mpi4py import MPI
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import chainermn


class ConcatenatedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)
        raise IndexError


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--communicator', type=str, default='hierarchical')
    parser.add_argument('--eval-iters', type=int, default=1000)
    parser.add_argument('--gpu', '-g', action='store_true')
    parser.add_argument('--log-iters', type=int, default=10)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--model-snapshot-iters', type=int, default=20000)
    parser.add_argument('--tot-iters', type=int, default=120000)
    args = parser.parse_args()

    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    n_workers = MPI.COMM_WORLD.Get_size()
    n_worker_iters = args.tot_iters // n_workers
    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(n_workers))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Iterations/worker: {}'.format(n_worker_iters))
        print('Total iterations: {}'.format(args.tot_iters))
        print()
        print('Using GPUs: {}'.format(args.gpu))
        print('Using {} communicator'.format(args.communicator))
        print('==========================================')

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_detection_label_names),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_detection_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
        #train_chain.to_gpu()

    if comm.rank == 0:
        train = TransformDataset(
            ConcatenatedDataset(
                VOCDetectionDataset(year='2007', split='trainval'),
                VOCDetectionDataset(year='2012', split='trainval')
            ),
            Transform(model.coder, model.insize, model.mean))
        test = VOCDetectionDataset(
            year='2007', split='test',
            use_difficult=True, return_difficult=True)
    else:
        train, test = None, None

    train = chainermn.scatter_dataset(train, comm)
    test = chainermn.scatter_dataset(test, comm)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(train_chain)

    # initial lr is set to 1e-3 by ExponentialShift
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater,
                               (n_worker_iters, 'iteration'),
                               args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger(
            [n_worker_iters // (3.0 / 2.0),
             n_worker_iters // (6.0 / 5.0)],
            'iteration')
        )

    if comm.rank == 0:
        trainer.extend(
            DetectionVOCEvaluator(
                test_iter, model, use_07_metric=True,
                label_names=voc_detection_label_names),
            trigger=(args.eval_iters, 'iteration'))

        trainer.extend(extensions.LogReport(trigger=(args.log_iters,
                                                     'iteration')))
        trainer.extend(extensions.observe_lr(), trigger=(args.log_iters,
                                                         'iteration'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'lr',
             'main/loss', 'main/loss/loc', 'main/loss/conf',
             'validation/main/map', 'elapsed_time']),
            trigger=(args.log_iters, 'iteration'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.snapshot(),
                       trigger=(args.model_snapshot_iters, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
            trigger=(args.model_snapshot_iters, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
