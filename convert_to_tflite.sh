#!/bin/bash

#block(name=convert_tflite, threads=4, memory=4000, gpus=1, hours=72)


model_name=$1
checkpoint_folder=$2
num_stages=$3
resolution=$4

model_file=$(cat ${checkpoint_folder}/checkpoint | grep model_checkpoint_path: | cut -d'"' -f2)


if [ ! -f ${checkpoint_folder}/checkpoint ]; then
	echo "Checkpoint file not found";
	echo ${checkpoint_folder}/checkpoint
	exit 1;
fi
if [ ! -f ${checkpoint_folder}/${model_file}.meta ]; then
	echo "model_file doesn't exist:";
	echo ${checkpoint_folder}/${model_file}.meta
	exit 1;
fi



python run_checkpoint.py --model $model_name \
    --resize $resolution \
    --num-stages $num_stages \
    --checkpoint=$checkpoint_folder

freeze_graph --input_graph=${checkpoint_folder}/graph_def_binary.pb \
    --input_checkpoint=${checkpoint_folder}/${model_file} \
    --input_binary=true \
    --output_node_names=Openpose/concat_stage${num_stages} \
    --output_graph=${checkpoint_folder}/graph_frozen.pb

tflite_convert    \
    --output_file=${checkpoint_folder}/model.tflite \
    --graph_def_file=${checkpoint_folder}/graph_frozen.pb \
    --input_arrays=image \
    --output_arrays=Openpose/concat_stage${num_stages}
