#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cascadia.py sequence \
/home/m/data3/Downloads/astral_data/PXD046386/1-5/mzml/02052023_Yeast_KO_3Th_2p5ms_KO1_rep01_20230503212731.mzML \
/home/m/data1/git/cascadia/pretrained_model/cascadia_astral_tuned.ckpt \
--outfile 386_mzml_result