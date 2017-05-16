#!/usr/bin/env bash


python train_faster_rcnn.py --gpu 0 --dataset voc07+voc_12 --iteration 210000 --out result/vgg16_voc07_voc12 --lr 0.001 --seed 0

python eval.py 0 result/vgg16_voc07_voc12/model
