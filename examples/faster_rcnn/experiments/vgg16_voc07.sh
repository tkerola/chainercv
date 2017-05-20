#!/usr/bin/env bash

GPU=$1
SEED=$2
OUT_DIR=result/vgg16_voc07/seed_$SEED

cd ..
python train_faster_rcnn.py --gpu $GPU --dataset voc07 --iteration 70000 --out $OUT_DIR --lr 0.001 --seed $SEED --step_size 50000

python eval.py 0 $OUT_DIR/snapshot_model.npz
