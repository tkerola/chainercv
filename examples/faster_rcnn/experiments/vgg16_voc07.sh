#!/usr/bin/env bash

cd ..
python train_faster_rcnn.py --gpu 0 --dataset voc07 --iteration 70000 --out result/vgg16_voc07 --lr 0.001 --seed 0

python eval.py 0 result/vgg16_voc07/model
