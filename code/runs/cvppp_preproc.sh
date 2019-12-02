#!/bin/bash

# name of the experiment
# (will create a directory with checkpoints and logs)
CFG=001_preproc
DIR=checkpoints/cvppp_${CFG}

source runs/utils.bash

check_rundir $DIR
mkdir $DIR

TRAIN_OPT="-learning_rate 1e-3 -batch_size 4 -iter_size 2 -learning_rate_decay 2e-5 -weightDecay 1e-4 -nEpochs 300"
LOSS_OPT="-predictAngles -predictFg -seq_length 21 -preproc_mode"
CMD="th main_preproc.lua -dataset cvppp -model_id $CFG $TRAIN_OPT $LOSS_OPT"

echo "Executing command: ${CMD}"

# saving the changes
# to reproduce the run
echo `git diff` > $DIR/001_train.diff
echo `git rev-parse HEAD` > $DIR/001_train.head

LOG=$DIR/001_train.log
echo "Logging: ${LOG}"
nohup $CMD > $LOG 2>&1 &

sleep 1
tail -f $LOG

exit 0;
