#!/bin/bash
train_dataset=./data/red_triangle_train_buf_augmented.pt
val_dataset=./data/red_triangle_val_buf_augmented.pt
name="red_triangle"
python train.py --name $name --train_dataset $train_dataset --val_dataset $val_dataset --architecture pick_to_place; python train.py --name $name --train_dataset $train_dataset --val_dataset $val_dataset --architecture pick_and_place

