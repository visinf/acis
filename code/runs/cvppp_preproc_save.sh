#!/bin/bash

EPOCH=300
CFG=611_preproc
DIR=checkpoints/cvppp_${CFG}
SEED=0

MODEL_OPT="-predictAngles -predictFg -seq_length 21 -batch_size 1 -preproc_epoch 300"
CMD="th main_preproc_save.lua -dataset cvppp -continue $CFG,$EPOCH $MODEL_OPT -preproc_mode -preproc_save"
echo $CMD

LOG=$DIR/001_preproc_save.log
nohup $CMD > $LOG 2>&1 &
echo "LOG: $LOG"

exit 0;
