#/bin/bash

#block(name=coco_b64_gpu2, threads=4, memory=4000, gpus=2, hours=72)


cmd="python tf_pose/train.py --imgpath /home/garbade/datasets/coco/images/ --datapath /home/garbade/datasets/coco/annotations/  --input-width 256 --input-height 384 --gpus 2 --checkpoint ./models/pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt"

echo $cmd

$cmd
