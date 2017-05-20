#!/usr/bin/env bash

# Train protocol follows
# https://github.com/endernewton/tf-faster-rcnn/blob/master/experiments/scripts/train_faster_rcnn.sh#L26

GPU=$1
SEED=$2
OUT_DIR=result/vgg16_voc0712/seed_$SEED

cd ..
python train_faster_rcnn.py --gpu $GPU --dataset voc07+voc12 --iteration 110000 --out $OUT_DIR --lr 0.001 --seed $SEED --step_size 80000

python eval.py 0 $OUT_DIR/snapshot_model.npz
