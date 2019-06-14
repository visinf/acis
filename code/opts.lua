--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'cvppp',    'Options: [cvppp | kitti]')
   cmd:option('-manualSeed', 128,        'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-gpu_id',     1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'deterministic',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-height',         530,    'Height of the resized images')
   cmd:option('-width',          500,    'Width of the resized images')
   cmd:option('-loadLast',     false,    'Load last seq_length instances (train time)')
   cmd:option('-loadLastTest', false,    'Load last seq_length instances (test time)')
   cmd:option('-seq_length',     21,     'Max. length of the sequence')
   cmd:option('-seq_length_test', 0,     'Length of the sequence at test time')
   cmd:option('-max_seq_length', 21,     'Max. length of the sequence')
   cmd:option('-vhigh',    false,        'Visualise Results in High Quality')
   cmd:option('-loss',        'bce',     'Mask Loss (bce|dice)')
   cmd:option('-optimActor', 'adam',     'Optimisation algorithm for actor')
   cmd:option('-optimCritic','adam',     'Optimisation algorithm for critic')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        4, 'number of data loading threads')
   ------------- Math MNIST options ------------------------
   cmd:option('-nums',            4, '# numbers in the expression')
   cmd:option('-maxNum',        100, 'Maximum number in the expression')
   cmd:option('-learnOp',     false, 'Learning the output prediction')
   cmd:option('-betaOp',          1, 'Trade-off between instance segmentation and regresion')
   cmd:option('-betaOrder',       1, 'Trade-off between instance segmentation and regresion')
   cmd:option('-gammaOp',         1, 'Trade-off between instance segmentation and regresion')
   ------------- Training options --------------------
   cmd:option('-nEpochs',     10000,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batch_size',     16,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-nSamples',       16,       'Number of samples to draw from N(0, I)')
   cmd:option('-noSampling',  false, 'Number of samples to draw from N(0, I)')
   cmd:option('-criticGT',    false, 'Critic has context from ground-truth')
   cmd:option('-criticTime',  false, 'Critic has timestep as input')
   cmd:option('-criticNoise',     0, 'Add noise for critic training')
   cmd:option('-actorNoise',      0, 'Add noise for actor training')
   cmd:option('-actorBN',     false, 'Switching the batchnorm for actor')
   cmd:option('-realNoise',       0, 'Inserting GT masks for the critic batch')
   cmd:option('-iter_size', 1, 'Quasi batch size')
   cmd:option('-summary_after', 16, 'Doing a summary of performance after this number of iterations')
   cmd:option('-checkpoint_after', 100, 'Saving the model (interval in epochs)')
   cmd:option('-validation_after', 10, 'Saving the model (interval in epochs)')
   cmd:option('-valTrain',    false, 'Run validation analysis on training')
   cmd:option('-dump',        false, 'Save the results (used with -checkOnly)')
   cmd:option('-semSeg',        false, 'Semantic segmentation mode (net predicts numClasses channels)')
   cmd:option('--save-stat', false, 'Save various statistic for analysis')
   ------------- Checkpointing options ---------------
   cmd:option('-model_id', '', 'Identification of the models')
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-continue', '', 'Snapshot path (e.g. VER/EPOCH)')
   cmd:option('-ctl', '', 'Snapshot path to ControlNet (e.g. VER/EPOCH)')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   cmd:option('-use_pretrained', false, 'Using pretrained model')
   ---------- Optimization options ----------------------
   cmd:option('-learning_rate',   1e-4, 'initial learning rate')
   cmd:option('-learning_rate_decay',   0.0, 'learning rate decay')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   cmd:option('-gamma', 1.0, 'Discount for Q-learning')
   cmd:option('-alpha',    0.99, 'Exp Run Avg coefficient for the Baseline')
   cmd:option('-betaCritic',    10, 'Critic update rate')
   cmd:option('-betaCNN',        1, 'CNN update rate')
   cmd:option('-betaCxt',        1, 'Context update rate')
   cmd:option('-betaBCE',        1, 'Weight of the BCE loss')
   cmd:option('-betaCTL',        1, 'Weight of the control loss')
   cmd:option('-betaGradCTL',    1, 'Weight of the control loss')
   cmd:option('-betaDec',      0.1, 'Weight of the control loss')
   cmd:option('-withLSTM',   false, 'Use RNN inputs [debug]')
   cmd:option('-subGt',      false, 'Substitute prediction with GT mask [debug]')
   cmd:option('-subLastMask',false, 'Substitute prediction with last GT mask [debug]')
   cmd:option('-reset_buffer', false, 'Constructing the buffer from scratch')
   cmd:option('-reset_context', false, 'Re-initialised context from scratch')
   cmd:option('-reset_critic', false, 'Re-initialised critic from scratch')
   cmd:option('-reset_control', false, 'Re-initialise control network')
   cmd:option('-critic_warmup', 512, 'Min. size of replay buffer')
   cmd:option('-critic_iter', 1, 'Min. size of replay buffer')
   cmd:option('-buffer_size', 1024, 'Max. size of replay buffer')
   cmd:option('-lambda', 0, 'Hyperparameter pondering the classification accuracy term.')
   cmd:option('-lambdaKL', 1, 'Scaler for KL divergence term')
   cmd:option('-sigma', 0, 'Scaler for KL divergence term')
   cmd:option('-actionY', 3, 'Action dimension (height)')
   cmd:option('-actionX', 3, 'Action dimension (width)')
   cmd:option('-contextType', 'max', 'Type of context (max | angles)')
   cmd:option('-contextDim', 3, 'Dimension of the learned memory')
   cmd:option('-decoder', '', 'Decoder network')
   cmd:option('-preproc', '', 'Preprocess network')
   cmd:option('-preproc_mode', false, 'Mode: training preprocess network')
   cmd:option('-preproc_save', false, 'Mode: saving results from preprocess network')
   cmd:option('-preproc_epoch', 15, 'Number of epochs for pre-processed augmentation')
   cmd:option('-dropRate', 0, 'Droput Rate')
   cmd:option('-numClasses', 1, 'Number of semantic classes')
   cmd:option('-numAngles', 8, 'Number of channels in the extmem')
   cmd:option('-numTestRuns', 1, 'Running validation on the held-out set # times')
   cmd:option('-numTrainRuns', 1, 'Running validation on the training set # times')
   cmd:option('-gradClip', 0, 'Gradient clipping for RNN')
   cmd:option('-bceCoeff', 1, 'BCE coefficient (used with -withBCELoss)')
   cmd:option('-withBCELoss', false, 'Use BCE Loss with the Actor-Critic')
   cmd:option('-normScore', false, 'Normalsed matching score')
   cmd:option('-enableContext', false, 'Propagate gradient to the action?')
   cmd:option('-withContext', false, 'Learning the context representation')
   cmd:option('-ignoreBg', false, 'Ignore background predictions [only with -predictFg]')
   cmd:option('-fgThreshold', 1e-3, 'Threshold for binarising FG prediction [only with -predictFg]')
   cmd:option('-ignoreMem', false, 'Ignore predictions of memory pixels (already predicted)')
   cmd:option('-ignoreMemThreshold', 0.5, 'Do not propagate errors on pixels with MEM set higher than that')
   cmd:option('-learnDecoder', false, 'Learn the decoder')
   cmd:option('-learnControl', false, 'Learning the control network')
   cmd:option('-oldControl', false, 'Specifies a standalone control net')
   cmd:option('-discreteLoss', false, 'Direct Loss Minisation (+AP)')
   cmd:option('-criticLoss', 'dice', 'Critic loss (dice|iou)')
   cmd:option('-maskNoise', 0, 'Random max() initialisation')
   cmd:option('-markov', false, 'Truncated RNN unrolling (only 2 steps)')
   cmd:option('-withVal', false, 'Include the validation set in the training')
   cmd:option('-markovCritic', false, 'Truncated unrolling for RNN critic (only 2 steps)')
   cmd:option('-maxEpochSize', 3200, 'Maximum epoch size')
   ---------- Model options ----------------------------------
   cmd:option('-checkOnly', false, 'No training; just validate and exit')
   cmd:option('-noEnc', false, 'Do not use encoder [pretraining-only]')
   cmd:option('-noise',     0, 'STD for white noise at bottleneck [pretraining-only]. 0 means no noise.')
   cmd:option('-logReward', false, 'Use log-scale rewards')
   cmd:option('-logRewardNorm', false, 'Use log-scale rewards log(1 + b*x)')
   cmd:option('-logRewardNormB', 0.01, 'Normaliser for -logRewardNorm')
   cmd:option('-invMask', false, 'Inverted Mask for prediction')
   cmd:option('-pyramidImage', false, 'Predict angles for pre-processing')
   cmd:option('-pyramidAngles', false, 'Predict angles for pre-processing')
   cmd:option('-pyramidFg', false, 'Predict foreground for pre-processing')
   cmd:option('-predictAngles', false, 'Predict angles for pre-processing')
   cmd:option('-predictFg', false, 'Predict foreground for pre-processing')
   cmd:option('-blockExtMem', false, 'Block external memory')
   cmd:option('-blockHidden', false, 'Block hidden LSTM/RNN state')
   cmd:option('-bottleneck', 'fc', 'Bottleneck architecture [fc|conv]')
   cmd:option('-sampler', 'vae', 'Sampler to use (vae|kernel)')
   cmd:option('-samplerAct', '', 'Use activation after sampler')
   cmd:option('-kernelSigmaL', 1, 'Location strength in the kernel sampler')
   cmd:option('-kernelSigmaF', 0.1, 'Feature strength in the kernel sampler')
   cmd:option('-reduction', 0.5, 'Reduction Factor after a Dense Block')
   cmd:option('-rnn_layers', 1, 'Number of layers of the rnn')
   cmd:option('-rnn_channels', 30, 'Number of channels of the rnn state')
   cmd:option('-latentSize', 16, 'Latent Size')
   cmd:option('-featureSize', 256, 'Feature Size')
   cmd:option('-rnnSize', 256, 'RNN Size')
   cmd:option('-criticMem', 3, 'Number of channels for critic memory')
   cmd:option('-criticDim', 32, 'Latent Size')
   cmd:option('-criticVer', 'simple', 'Version of the critic architecture (simple | densenet)')
   cmd:option('-criticType', 'fc', 'Type of the critic global pooling (max | avg | softmax_cmul | fc)')
   cmd:option('-contextVer', 'simple', 'Version of the context net')
   cmd:option('-actorVer', 'simple', 'Version of the actor architecture (simple | densenet)')
   cmd:option('-decVer', 'simple', 'Version of the decoder architecture (simple | resnet | densenet | softmax)')
   cmd:option('-actType', 'simple', 'Version of the actor bottleneck (simple | softmax | softmax_cmul)')
   cmd:option('-decVerMod', 'none', 'Sub-Version of the decoder architecture (extra | none)')
   cmd:option('-decVerUp', 'deconv', 'Upsampling version of the decoder architecture (deconv | nn | bilinear)')
   cmd:option('-rewardVer', 'match', 'Version of the immediate reward (greedy | match)')
   cmd:option('-grad', 'ac', 'Type of the gradient (ac | reinforce | fixed)')
   cmd:option('-orderby', 'hung', 'If grad == fixed, order by [hung|size|greedy]')
   cmd:option('-matchNoise', 0.0, 'If > 0, adds Gaussian noise to the matching matrix')
   cmd:option('-hyperFile', '', 'filename to write the result for Bayesian optimisation')

   --
   -- crayon visualisation
   --
   cmd:option('-crayonHost', 'localhost', 'Address to crayon server')
   cmd:option('-crayonPort',        6039, 'Port to crayon server')


   cmd:text()

   -- parse input params
   opt = cmd:parse(arg)
   opt.tensorType = 'torch.CudaTensor'
   opt.optMemory = 3
   opt.imageCh = 3

   if opt.dataset == 'cvppp' then
     print('Using CVPPP dataset')
     if opt.withVal then
       print('>>> Validation Set Included <<<')
       opt.data = 'data/cvppp/A1_ALL/'
       opt.gen = 'gen_all'
     else
       if opt.preproc_mode then
         opt.data = 'data/cvppp/A1_RAW/'
       else
         opt.data = 'data/cvppp/A1_AUG/'
       end
     end

     if opt.actorVer == 'resnet_dense' then
       opt.actionX = 14
       opt.actionY = 14
     else
       opt.actionX = 7
       opt.actionY = 7
     end

     opt.Size = 224
     opt.imHeight = 224
     opt.imWidth = 224
     opt.Crop = 224
     opt.xSize = 224
     opt.ySize = 237 -- 224
     opt.featureSize = 256
     opt.numClasses = 2
     opt.growthRate = 32
     opt.maxEpochSize = 256
     opt.max_seq_length = 21
   elseif opt.dataset == 'kitti' then
     print('Using KITTI dataset')
     if opt.preproc_mode then
       opt.data = 'data/kitti'
     else
       opt.data = 'data/kitti_aug'
     end
     opt.actionX = 24
     opt.actionY = 8
     opt.Size = 256
     opt.imHeight = 256 --512
     opt.imWidth = 768--1024
     opt.ySize = 256 -- 370
     opt.xSize = 768 -- 1224
     opt.latentSize = 64
     opt.featureSize = 512 --256
     opt.numClasses = 2
     opt.growthRate = 16
     opt.dropRate = 0
     opt.max_seq_length = 16
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.seq_length_test == 0 then
    opt.seq_length_test = opt.max_seq_length
   end

   opt.loadSeqLength = opt.seq_length
   opt.loadSeqLengthTest = opt.seq_length_test

   if opt.loadLast then
     opt.loadSeqLength = -opt.loadSeqLength
   end

   if opt.loadLastTest then
     opt.loadSeqLengthTest = -opt.loadSeqLengthTest
   end

   print('--        Dataset: ' .. opt.dataset)
   print('--  learning rate: ' .. opt.learning_rate)
   print('--        Q-Gamma: ' .. opt.gamma)
   print('--    Latent Size: ' .. opt.latentSize)
   print('--     Batch Size: ' .. opt.batch_size)
   print('---------------------------------')
   print(opt)
   print('---------------------------------')

   print(string.format('--         Action: %dx%d', opt.actionY, opt.actionX))

   local model_id = opt.model_id or os.date('%Y%m%d%H%M%S', os.time())
   opt.config_id = string.format('%s_%s', opt.dataset, model_id)
   opt.save_dir = opt.save
   opt.save = paths.concat(opt.save, opt.config_id)
   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('Options: Unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   local cxtManager = ContextManager(opt)
   if opt.contextType == 'max' then
     -- applying pixelwise max with the memory and the mask
     opt.cxtSize = 1
   elseif opt.contextType == 'angles' then
     -- merging the mask in 2-dimensional angle representation (x,y)
     opt.cxtSize = 2
   else
		 cmd:error('Options: Unknown context representation: ' .. opt.contextType)
   end

   return opt
end

return M
