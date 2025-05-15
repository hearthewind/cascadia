#!/bin/bash

python my_cascadia.py sequence \
/home/m/Temp/46386_1-5_1/02052023_Yeast_KO_3Th_2p5ms_KO1_rep01_20230503212731_1.csv \
/home/m/data1/git/cascadia/pretrained_model/cascadia_astral_tuned.ckpt \
--score_threshold 0.0 \
--width 10 \
--outfile 386_myloader_result