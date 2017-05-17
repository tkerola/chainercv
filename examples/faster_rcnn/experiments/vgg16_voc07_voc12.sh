#!/usr/bin/env bash

# Train protocol follows
# https://github.com/endernewton/tf-faster-rcnn/blob/master/experiments/scripts/train_faster_rcnn.sh#L26

cd ..
python train_faster_rcnn.py --gpu 0 --dataset voc07+voc12 --iteration 110000 --out result/vgg16_voc07_voc12 --lr 0.001 --seed 0 --step_size 80000

python eval.py 0 result/vgg16_voc07_voc12/model
