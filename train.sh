#!/usr/bin/env bash
file_list='data/train_data_68/list.txt'
test_list='data/test_data_68/list.txt'
save_model=models2/wflw_moruhard_grow_68_Res50
pretrained_model=
logs=./models2/log0.txt
lr=0.0001
num_label=68
image_size=128

CUDA_VISIBLE_DEVICES=0 python -u train_model.py --model_dir=${save_model} \
                                                --learning_rate=${lr} \
												--file_list=${file_list} \
												--test_list=${test_list} \
												--pretrained_model=${pretrained_model} \
                                                --lr_epoch='20,50,100,300,400' \
				                				--level=L1 \
				                				--image_size=${image_size} \
				                				--image_channels=3 \
				                				--batch_size=128 \
					        					--max_epoch=500 \
												--num_label=${num_label} \
                                                > ${logs} 2>&1 &
tail -f ${logs}
