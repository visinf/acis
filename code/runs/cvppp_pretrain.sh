#!/bin/bash

CFG=001_pretrain
DIR=checkpoints/cvppp_${CFG}
SEED=0

mkdir $DIR

#################################
# Pretraining (decoder)
#################################

### Pre-training
MODEL_OPTS="-rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple"
DATA_OPTS="-seq_length -1 -loadLast"
TRAIN_OPTS="-learning_rate 2e-4 -lambdaKL 1 -checkpoint_after 200 -weightDecay 1e-4 -validation_after 50 -grad fixed -batch_size 8 -iter_size 1 -summary_after 4 -gradClip 1 -numTestRuns 8 -numTrainRuns 8 -valTrain -nEpochs 400 -markov"
th pretrain_main.lua  -dataset cvppp -model_id $CFG $MODEL_OPTS $DATA_OPTS $TRAIN_OPTS 

### Pre-training: reduce LR to 2e-5
MODEL_OPTS="-rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple"
DATA_OPTS="-seq_length -1 -loadLast"
TRAIN_OPTS="-learning_rate 2e-5 -lambdaKL 1 -checkpoint_after 200 -weightDecay 1e-4 -validation_after 50 -grad fixed -batch_size 8 -iter_size 1 -summary_after 4 -gradClip 1 -numTestRuns 8 -numTrainRuns 8 -valTrain -nEpochs 600 -markov"
CHECKPOINT="-continue ${CFG},400"
th pretrain_main.lua -dataset cvppp -model_id $CFG $CHECKPOINT $MODEL_OPTS $DATA_OPTS $TRAIN_OPTS 

### Pre-training: reduce LR to 2e-6
MODEL_OPTS="-rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple"
DATA_OPTS="-seq_length -1 -loadLast"
TRAIN_OPTS="-learning_rate 2e-6 -lambdaKL 1 -checkpoint_after 200 -weightDecay 1e-4 -validation_after 50 -grad fixed -batch_size 8 -iter_size 1 -summary_after 4 -gradClip 1 -numTestRuns 8 -numTrainRuns 8 -valTrain -nEpochs 800 -markov"
CHECKPOINT="-continue ${CFG},600"
th pretrain_main.lua -dataset cvppp -model_id $CFG $CHECKPOINT $MODEL_OPTS $DATA_OPTS $TRAIN_OPTS 


#################################
# Training on sequence length 1
#################################

CFG_DEC=001_pretrain,800
CFG=001_init
DIR=checkpoints/cvppp_${CFG}
SEED=0

mkdir $DIR

### Training, init: Length 1
TRAIN_OPTS="-learning_rate 5e-4 -lambdaKL 0.1 -optimActor adam -learning_rate_decay 0.0 -gradClip 1 -grad fixed -batch_size 8 -weightDecay 1e-4"
LOG_OPTS="-checkpoint_after 400 -validation_after 100 -summary_after 4 -numTestRuns 8 -numTrainRuns 8 -valTrain -nEpochs 400"
MODEL_OPTS="-markov -rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple -contextType max"
DATA_OPTS="-dataset cvppp -seq_length 1 -seq_length_test 1 -loadLast -loadLastTest"
th main.lua -dataset cvppp -model_id $CFG -decoder $CFG_DEC $TRAIN_OPTS $LOG_OPTS $MODEL_OPTS $DATA_OPTS -manualSeed $SEED

### Training, with sampling: Length 1
TRAIN_OPTS="-learning_rate 5e-4 -lambdaKL 0 -optimActor adam -learning_rate_decay 0.01 -gradClip 1 -grad fixed -batch_size 8 -weightDecay 1e-4 -noSampling"
LOG_OPTS="-checkpoint_after 600 -validation_after 100 -summary_after 4 -numTestRuns 8 -numTrainRuns 8 -valTrain -nEpochs 600"
MODEL_OPTS="-markov -rnn_layers 2 -rnnSize 512 -latentSize 16 -predictFg -predictAngles -pyramidImage -pyramidFg -pyramidAngles -oldControl -bottleneck fc -actorVer simple -contextType max"
DATA_OPTS="-dataset cvppp -seq_length 1 -seq_length_test 1 -loadLast -loadLastTest"
th main.lua -dataset cvppp -model_id $CFG -continue $CFG,400 -decoder $CFG_DEC $TRAIN_OPTS $LOG_OPTS $MODEL_OPTS $DATA_OPTS -manualSeed $SEED


exit 0;
