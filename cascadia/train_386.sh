#!/bin/bash

python my_cascadia.py train \
/home/m/data3/Raw_Msgp_for_DeepNovo/46386_train_valid/valid.csv \
/home/m/data3/Raw_Msgp_for_DeepNovo/46386_train_valid/valid.csv \
--model /home/m/data1/git/cascadia/pretrained_model/386model.ckpt \
--width 10