#/bin/bash

#block(name=01_b48_m2_smll_stg5_rsm, threads=4, memory=4000, gpus=1, hours=72)


python tf_pose/train.py \
	--imgpath /home/garbade/datasets/coco/images/ \
	--datapath /home/garbade/datasets/coco/annotations/  \
	--input-width 256 \
	--input-height 384 \
	--batchsize 48 \
	--gpus 1 \
	--model mobilenet_v2_small \
	--num-stages 5 \
	--checkpoint ./models/train/aisc01_mobilenet_v2_small_stage5/model_latest-180000

#	--checkpoint ./models/pretrained/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt



