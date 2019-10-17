#/bin/bash

#block(name=m2_075_b48_gpu4, threads=4, memory=4000, gpus=4, hours=144)


cmd="python tf_pose/train.py \
         --imgpath /home/garbade/datasets/coco/images/ \
         --datapath /home/garbade/datasets/coco/annotations/ \
         --input-height 384 \
         --input-width 256  \
         --batchsize 48 \
         --gpus 4  \
         --model mobilenet_v2_0.75 \
         --checkpoint ./models/pretrained/mobilenet_v2_0.75_224/mobilenet_v2_0.75_224.ckpt"

echo $cmd

$cmd
