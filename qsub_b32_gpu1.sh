#/bin/bash

#block(name=coco_b32_gpu1, threads=4, memory=4000, gpus=1, hours=72)


python tf_pose/train.py \
	--imgpath /home/garbade/datasets/coco/images/ \
	--datapath /home/garbade/datasets/coco/annotations/  \
	--input-width 256 \
	--input-height 384 \
	--batchsize 32 \
	--gpus 1 \
	--model mobilenet_v2_0.75

#	--checkpoint ./models/pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt



