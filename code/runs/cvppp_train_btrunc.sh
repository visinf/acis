#!/bin/bash

CFG_DEC=001_pretrain,800
CFG_INIT=001_init,600

CFG=001_btrunc
DIR=checkpoints/cvppp_${CFG}
SEED=0

CRAYON="-crayonPort 6042"
MODEL_OPTS="-markov -rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple -contextType max"
TRAIN_OPTS="-learning_rate 5e-4 -lambdaKL 0 -optimActor adam -learning_rate_decay 0.0 -gradClip 1 -grad fixed -batch_size 8 -weightDecay 1e-5 -noSampling -matchNoise 0.0"
LOG_OPTS="-checkpoint_after 2000 -validation_after 100 -summary_after 4 -numTestRuns 1 -numTrainRuns 1 -valTrain -nEpochs 10000"
DATA_OPTS="-dataset cvppp -seq_length 5 -seq_length_test 21 -loadLast"

source runs/utils.bash

check_rundir $DIR
mkdir $DIR

git diff > $DIR/001_train.diff
nohup th main.lua -dataset cvppp -model_id $CFG -continue $CFG_INIT -decoder $CFG_DEC $TRAIN_OPTS $LOG_OPTS $MODEL_OPTS $DATA_OPTS > $DIR/001_train.log 2>&1 &
echo "LOG: $DIR/001_train.log"

exit 0;
