#!/bin/bash

CFG=611_preproc
DIR=checkpoints/cvppp_${CFG}
SEED=0

source runs/utils.bash

check_rundir $DIR
mkdir $DIR

TRAIN_OPT="-learning_rate 1e-3 -batch_size 4 -iter_size 2 -learning_rate_decay 2e-5 -weightDecay 1e-4 -nEpochs 300"
LOSS_OPT="-predictAngles -predictFg -seq_length 21 -preproc_mode"
CMD="th main_preproc.lua -dataset cvppp -model_id $CFG $TRAIN_OPT $LOSS_OPT"
echo $CMD

LOG=$DIR/001_train.log
nohup $CMD > $LOG 2>&1 &

exit 0;
