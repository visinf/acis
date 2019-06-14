local vis = require 'utils/vis'

local analysis = require 'analysis'
local hungarian = require 'hungarian'
require 'utils/instance_map'
local flow = require 'utils/flowExtensions'
local seq = require 'pl/seq'

local M = {}
local Trainer = torch.class('acis.Trainer', M)

function mean(t)
  local sum = 0
  local count = 0

  for k,v in pairs(t) do
    if type(v) == 'number' then
      sum = sum + v
      count = count + 1
    end
  end

  return (sum / count)
end

function gradRGB(mask, grad)
  assert(mask:dim() == 2 and grad:dim() == 2)
  local h, w = unpack(mask:size():totable())
  local h_, w_ = unpack(grad:size():totable())
  assert(h == h_ and w == w_)

  function splitFunc(x, idx)
    local xAbs = torch.abs(x[idx])
    local maxVal = xAbs:max()
    local out = torch.zeros(h, w)
    out[idx] = xAbs
    return out:div(maxVal + (maxVal == 0 and 1 or 0))
  end

  local posVal = torch.gt(grad, 0)
  local negVal = torch.lt(grad, 0)

  local posMap = splitFunc(grad, posVal)
  local negMap = splitFunc(grad, negVal)

  local rgb = torch.zeros(3, h, w)
  rgb[1]:copy(posMap)
  rgb[2]:copy(mask)
  rgb[3]:copy(negMap)

  return rgb
end

function Trainer:__init(model, criterion, opt, optimState, loggers)
  self.model = model
  self.criterion = criterion

  self.optimState = optimState
  if not self.optimState then
    print("[Initialising optimState]")
    self.optimState = {}
    self.optimState.Actor = {learningRate = opt.learning_rate,
                             weightDecay = opt.weightDecay}
    
    if opt.learnDecoder then
      self.optimState.Decoder = {learningRate = opt.learning_rate,
                                 weightDecay = opt.weightDecay}
    end

    if opt.grad == 'ac' then
      self.optimState.Critic   = {learningRate = opt.betaCritic * opt.learning_rate, weightDecay = opt.weightDecay} --, momentum = 0.9, nesterov = true, dampening = 0}
    end

    if opt.learnControl then
      self.optimState.Control = {learningRate = opt.betaCTL * opt.learning_rate, weightDecay = opt.weightDecay}
    end
  else

    print("[Updating optimState]")

    self.optimState.Actor.learningRate = opt.betaCNN * opt.learning_rate
    self.optimState.Actor.learningRateDecay = nil --opt.learning_rate_decay
    self.optimState.Actor.weightDecay = opt.weightDecay

    if opt.learnControl then
      self.optimState.Control.learningRate = opt.betaCTL * opt.learning_rate
      self.optimState.Control.weightDecay = opt.weightDecay
    end

    if opt.learnDecoder then
      if self.optimState.Decoder then
        self.optimState.Decoder.learningRate = opt.betaDec * opt.learning_rate
        self.optimState.Decoder.weightDecay = opt.weightDecay
      else
        self.optimState.Decoder = {learningRate = opt.betaDec * opt.learning_rate,
                                   weightDecay = opt.weightDecay}
      end
    end

    if opt.grad == 'ac' then
      self.optimState.Critic.learningRate = opt.betaCritic * opt.learning_rate
      self.optimState.Critic.weightDecay = opt.weightDecay
    end
  end

  print('optimState.Actor = ', self.optimState.Actor)

  if opt.learnDecoder then
    print('optimState.Decoder = ', self.optimState.Decoder)
  end

  if opt.learnControl then
    print('optimState.Control = ', self.optimState.Control)
  end

  if opt.grad == 'ac' then
    print('optimState.Critic = ', self.optimState.Critic)
  end

  self.opt = opt
  print('Using ' .. opt.optimActor .. ' for the actor')
  print('Using ' .. opt.optimCritic .. ' for the critic')

  self.optimMethod = optim[opt.optimActor]
  self.optimMethodCritic = optim[opt.optimCritic]
  
  -- initialising logging with crayon
  self.loggers = loggers

  -- the initial state of the cell/hidden states
  self.init_state = {}
  self.init_state_batch = {}
  self.rnnStatesLast = {}
  local h_init = torch.zeros(1, opt.rnnSize)
  local h_init_batch = torch.zeros(opt.batch_size, opt.rnnSize)
  if opt.nGPU > 0 then
    h_init = h_init:cuda()
    h_init_batch = h_init_batch:cuda()
  end

  for L=1,2*opt.rnn_layers do
    table.insert(self.init_state, h_init:clone())
    table.insert(self.init_state_batch, h_init_batch:clone())
    table.insert(self.rnnStatesLast, h_init_batch:clone())
  end

  self.actorParamsX, self.actorGradX = self.model.Actor:getParameters()
  self.model.Actor:zeroGradParameters()
  print('Actor has > ' .. self.actorParamsX:nElement() .. ' < parameters')

  if opt.grad == 'ac' then
    self.criticParamsX,   self.criticGradX   = self.model.Critic:getParameters()
    print('Critic has > ' .. self.criticParamsX:nElement() .. ' < parameters')
  end

  if opt.learnControl then
    self.controlParamsX,   self.controlGradX   = self.model.Control:getParameters()
    print('ControlNet has > ' .. self.controlParamsX:nElement() .. ' < parameters')
    self.model.Control:zeroGradParameters()
    self.model.Control:training()
  end

  ---- Saving the RNN states for unrolling
  self.modelActorStates = {}
  if not opt.markov then
    for t = 1,self.opt.seq_length do
      table.insert(self.modelActorStates, self.model.Actor:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_var'))
    end
  end

  if opt.learnDecoder then
    self.modelDecoderStates = {}
    if not opt.markov then
      for t = 1,self.opt.seq_length do
        table.insert(self.modelDecoderStates, self.model.Decoder:clone('weight', 'bias', 'gradWeight', 'gradBias'))
      end
    end
    if opt.learnDecoder then
      self.decoderParamsX, self.decoderGradX = self.model.Decoder:getParameters()
      self.model.Decoder:zeroGradParameters()
    end
  end

  if self.opt.grad == 'ac' then
    self.gradTDZero = torch.Tensor(opt.batch_size, 1):zero()
    self.maxGrad = torch.Tensor(opt.batch_size, 1):fill(-1)

    if self.opt.nGPU > 0 then 
      self.maxGrad = self.maxGrad:cuda()
      self.gradTDZero = self.gradTDZero:cuda()
    end

    self.model.Critic:training()
    self.model.Critic:zeroGradParameters()

    if opt.criticVer == 'lstm' then
      self.criticInit = torch.zeros(opt.batch_size, 512)
      self.criticInit1 = torch.zeros(1, 512)

      if self.opt.nGPU > 0 then
        self.criticInit = self.criticInit:cuda()
        self.criticInit1 = self.criticInit1:cuda()
      end

      self.gradCritic = {self.maxGrad:clone(), self.criticInit:clone(), self.criticInit:clone()}
      self.cRnnGrad = {self.criticInit:clone(), self.criticInit:clone()}

      self.modelCriticStates = {}
      for t = 1,self.opt.seq_length do
        table.insert(self.modelCriticStates, self.model.Critic:clone('weight', 'bias', 'gradWeight', 'gradBias'))
        self.modelCriticStates[t]:training()
      end
    elseif opt.criticVer == 'irnn' or opt.criticVer == 'irnn_pool' then
      self.criticInit = torch.zeros(opt.batch_size, opt.criticMem, opt.imHeight, opt.imWidth)
      self.criticInit1 = torch.zeros(1, opt.criticMem, opt.imHeight, opt.imWidth)

      if self.opt.nGPU > 0 then
        self.criticInit = self.criticInit:cuda()
        self.criticInit1 = self.criticInit1:cuda()
      end

      self.gradCritic = {self.maxGrad:clone(), self.criticInit:clone()}
      self.cRnnGrad = self.criticInit:clone()

      self.modelCriticStates = {}
      if opt.markovCritic then
        table.insert(self.modelCriticStates, self.model.Critic:clone('weight', 'bias', 'gradWeight', 'gradBias'))
        table.insert(self.modelCriticStates, self.model.Critic:clone('weight', 'bias', 'gradWeight', 'gradBias'))
      else
        for t = 1,self.opt.seq_length do
          table.insert(self.modelCriticStates, self.model.Critic:clone('weight', 'bias', 'gradWeight', 'gradBias'))
          self.modelCriticStates[t]:training()
        end
      end
    else
      self.gradCritic = self.maxGrad:clone()
    end
  end

  if self.opt.enableContext then
    self.sigmoid = nn.Sigmoid():cuda()
  end

	self.maskSum = torch.Tensor(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
 	self.extMem = torch.Tensor(self.opt.batch_size, opt.cxtSize, self.opt.imHeight, self.opt.imWidth)

  if self.opt.nGPU > 0 then
    self.extMem = self.extMem:cuda()
		self.maskSum = self.maskSum:cuda()
  end

  self.rnnGrad = {}
  for tt = 1,#self.init_state do
    self.rnnGrad[tt] = torch.zeros(opt.batch_size, opt.rnnSize)
    if self.opt.nGPU > 0 then
      self.rnnGrad[tt] = self.rnnGrad[tt]:cuda()
    end
  end

  self.maskZero = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
  if self.opt.nGPU > 0 then self.maskZero = self.maskZero:cuda() end

  self.clsGrad = torch.zeros(opt.batch_size, opt.numClasses)
  if self.opt.nGPU > 0 then self.clsGrad = self.clsGrad:cuda() end

  self.ctlZero = torch.zeros(opt.batch_size, 1)
  if self.opt.nGPU > 0 then self.ctlZero = self.ctlZero:cuda() end

  -- zero class (termination)
  self.ctlStop = torch.ones(opt.batch_size)
  if self.opt.nGPU > 0 then self.ctlStop = self.ctlStop:cuda() end

  self.ctlRun = torch.Tensor(opt.batch_size):fill(2)
  if self.opt.nGPU > 0 then self.ctlRun = self.ctlRun:cuda() end

  self.actionZero = torch.zeros(opt.batch_size, opt.latentSize, opt.actionY, opt.actionX)
  if self.opt.nGPU > 0 then self.actionZero = self.actionZero:cuda() end

  if self.opt.predictFg then
    self.Fg = torch.Tensor(1, 1, opt.imHeight, opt.imWidth)
    self.binFg = torch.Tensor(1, 1, opt.imHeight, opt.imWidth)
    if self.opt.nGPU > 0 then
      self.Fg = self.Fg:cuda()
      self.binFg = self.binFg:cuda()
    end
  end

  if self.opt.predictAngles then
    self.angleMask = torch.Tensor(1, opt.numAngles, opt.imHeight, opt.imWidth)
    if self.opt.nGPU > 0 then self.angleMask = self.angleMask:cuda() end
  end

  self:resetInfo()

  -- preparing the IoU thresholds
  self.avgs = torch.Tensor({0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95})

  -- the batch buffer
  -- will contain iter_size sequences
  -- each with size batch_size
  self:allocateBatch()

  -- Timer
  self.timer = torch.Timer()

  if self.opt.criticLoss == 'dice' then
    self.softLoss = function(a, b) return self:softDice(a, b) end
  elseif self.opt.criticLoss == 'iou' then
    self.softLoss = function(a, b) return self:softIoU(a, b) end
  elseif self.opt.criticLoss == 'bce' then
    self.softLoss = function(a, b) return 1-1e-4*self.criterion.BCE:forward(a, b) end
  end

  if self.opt.discreteLoss then
    if self.opt.criticLoss == 'dice' then
      self.hardLoss = function(a, b) return self:dice(a, b) end
    elseif self.opt.criticLoss == 'iou' then
      self.hardLoss = function(a, b) return self:AP(a, b) end
    end
  end

  -- parameter update routines
  self.fevalActor = function()
    return -1, self.actorGradX
  end

  -- parameter update routines
  self.fevalContext = function()
    return -1, self.contextGradX
  end

  self.fevalDecoder = function()
    return -1, self.decoderGradX
  end

  self.fevalControl = function()
    return self.criterion.CE.output, self.controlGradX
  end

  if opt.grad == 'ac' then
    self.fevalCritic = function()
      return self.criterion.TD.output, self.criticGradX
    end
  end

  self.batchMatchIndex = nn.BatchMatchIndex(opt.orderby, opt.max_seq_length, opt.matchNoise)
  self.bestScoreIndex = nn.BestScoreIndex()

  self.getMaskAngles = nn.MaskAngles(opt.imHeight, opt.imWidth)

  local cxtManager = ContextManager(opt)
  local cxtManagerNonBatch = ContextManager(opt, true)
  if opt.contextType == 'max' then
    -- applying pixelwise max with the memory and the mask
    -- dataloader uses the same 
	  self.cxtUpdate = function(x,y,z) cxtManager:max(x,y,z) end
    self.cxtUpdateDataloader = function(x,y,z) cxtManagerNonBatch:max(x,y,z)  end
  elseif opt.contextType == 'angles' then
    -- merging the mask in 2-dimensional angle representation (x,y)
    cxtManager:copyCuda()
    self.cxtUpdate = function(x,y,z) cxtManager:angles(x,y,z) end
	  self.cxtUpdateDataloader = function(x,y,z) cxtManagerNonBatch:angles(x,y,z) end
  end

  self:resetBuffer()

end

function Trainer:trainBaseline(epoch, dataloader)

  local info = {}

  local stepActor
  if self.opt.markov then
    stepActor = function(info, b, noisy, len) return self:stepActorMarkov(info, b, true, {}, noisy, len) end
  else
    stepActor = function(info, b, noisy, len) return self:stepActor(info, b, true, {}, noisy, len) end
  end

  self.iter = dataloader:size() * (epoch - 1)

  local n = 0
  for sample in dataloader:run(opt.loadSeqLength, self.cxtUpdateDataloader) do

    local time = sys.clock()

    ---------------- Preparing the sample ------------------

    if self.opt.nGPU > 0 then
      sample.input = sample.input:cuda()
      sample.target = sample.target:cuda()

      if self.opt.predictAngles then sample.angles = sample.angles:cuda() end
      if self.opt.predictFg then sample.fg = sample.fg:cuda() end
    end

    self:resetStat(info)
    self:playBatch(info, sample, 0)

    --------------------- Learning Part --------------------
    self.model.Actor:training()
    for t = 1,#self.modelActorStates do self.modelActorStates[t]:training() end
    if self.opt.learnControl then self.model.Control:training() end

    stepActor(info, sample, 0)

    if self.iter % self.opt.iter_size == 0 then
      self:updateActor()
    end

    self:updateInfo(info, time)

    if self.iter % self.opt.summary_after == 0 then
      local cls_loss = mean(self.summaryClsLoss)
      io.write(string.format('%08.1f [%04d/%03d] ', self.timer:time().real, epoch, n))
      io.write(string.format('%5.3f ms | TD %4.3e / BCE %4.3e / CLS %4.3e / KL %3.2e / Mean %4.3e / Var %4.3e / P %4.3e / IoU %4.3f / Reward %4.3f / Dice %4.3f / AP %4.3f\n', 
                                  mean(self.summaryTime),
                                  mean(self.summaryTDLoss),
                                  mean(self.summaryBCELoss),
                                  cls_loss,
                                  mean(self.summaryKLLoss),
                                  mean(self.summaryKLMean),
                                  mean(self.summaryKLVar),
                                  mean(self.summaryPLoss),
                                  mean(self.summaryIoU),
                                  mean(self.summaryReward),
                                  mean(self.summaryDice),
                                  mean(self.summaryAP)))

      if self.loggers.train then
        self.loggers.train:add_scalar_value("data/reward", mean(self.summaryReward), -1, self.iter)
        self.loggers.train:add_scalar_value("data/dice",   mean(self.summaryDice),   -1, self.iter)
        self.loggers.train:add_scalar_value("data/iou",    mean(self.summaryIoU),    -1, self.iter) 

        -- add if not nan
        if cls_loss == cls_loss then 
          self.loggers.train:add_scalar_value("data/cls", cls_loss, -1, self.iter)
        end

        self.loggers.train:add_scalar_value("data/mask_loss", mean(self.summaryBCELoss), -1, self.iter)
        self.loggers.train:add_scalar_value("data/kld_loss",  mean(self.summaryKLLoss),  -1, self.iter)

        self.loggers.train:add_scalar_value("data/kl_mean",   mean(self.summaryKLMean), -1, self.iter)
        self.loggers.train:add_scalar_value("data/kl_var",    mean(self.summaryKLVar),  -1, self.iter)
      end

      io.flush()
      self:clearStates()
      self:resetInfo()
    end

    self.iter = self.iter + 1
    n = n + 1
  end
end


function Trainer:train(epoch, dataloader, dataloaderCritic)

  local info = {}

  local stepActor, stepCritic
  if self.opt.markov then
    stepActor = function(info, b, noisy) return self:stepActorMarkov(info, b, true, {}, noisy) end
  else
    stepActor = function(info, b, noisy) return self:stepActor(info, b, true, {}, noisy) end
  end

  if self.opt.grad == 'ac' then
    stepCritic = function(info, b) return self:stepCritic(info, b) end
  end

  self.iter = dataloader:size() * (epoch - 1)
  local criticTurn = false
  if self.opt.grad == 'ac' then
    criticTurn = self.iter < self.opt.critic_warmup or (self.iter % self.opt.critic_iter == 0)
  end

  local n = 0
  for sample, xsample in seq.zip(dataloader:run(opt.loadSeqLength, self.cxtUpdateDataloader), dataloaderCritic:run(opt.loadSeqLength, self.cxtUpdateDataloader)) do

    local time = sys.clock()

    ---------------- Preparing the sample ------------------
    
    self:resetStat(info)

    --------------------- Learning Part --------------------
    if criticTurn then
      self:copyGPU(xsample)
      self:playBatch(info, xsample, self.opt.criticNoise, self.opt.max_seq_length)

      stepCritic(info, xsample)

      if self.iter % self.opt.iter_size == 0 then
        self:updateCritic()
      end
    else
      self:copyGPU(sample)
      self:playBatch(info, sample, 0)

      self.model.Actor:training()
      for t = 1,#self.modelActorStates do self.modelActorStates[t]:training() end
      if self.opt.learnControl then self.model.Control:training() end

      stepActor(info, sample, self.opt.actorNoise)

      if (self.iter / self.opt.critic_iter) % self.opt.iter_size == 0 then
        self:updateActor()
      end
    end

    self:updateInfo(info, time)

    if self.iter % self.opt.summary_after == 0 then
      io.write(string.format('%08.1f [%04d/%03d] ', self.timer:time().real, epoch, n))
      io.write(string.format('%5.3f ms | TD %4.3e / BCE %4.3e / CLS %4.3e / KL %3.2e / Mean %4.3e / Var %4.3e / P %4.3e / IoU %4.3f / R %4.3e / Dice %4.3f / AP %4.3f\n', 
                                  mean(self.summaryTime),
                                  mean(self.summaryTDLoss),
                                  mean(self.summaryBCELoss),
                                  mean(self.summaryClsLoss),
                                  mean(self.summaryKLLoss),
                                  mean(self.summaryKLMean),
                                  mean(self.summaryKLVar),
                                  mean(self.summaryPLoss),
                                  mean(self.summaryIoU),
                                  info.sumReward / info.numReward,
                                  mean(self.summaryDice),
                                  mean(self.summaryAP)))

      local logIfNumber = function(name, summary)
        local meanVal = mean(summary)
        if meanVal == meanVal then
          self.loggers.train:add_scalar_value(name, meanVal, -1, self.iter)
        end
      end

      if self.loggers.train then
        logIfNumber("data/reward",    self.summaryReward)
        logIfNumber("data/dice",      self.summaryDice)
        logIfNumber("data/iou",       self.summaryIoU)
        logIfNumber("data/cls",       self.summaryClsLoss)
        logIfNumber("data/mask_loss", self.summaryBCELoss)
        logIfNumber("data/kld_loss",  self.summaryKLLoss)
        logIfNumber("data/critic_td", self.summaryTDLoss)
        logIfNumber("data/kl_mean",   self.summaryKLMean)
        logIfNumber("data/kl_var",    self.summaryKLVar)
      end

      io.flush()
      self:clearStates()
      self:resetInfo()
    end

    self.iter = self.iter + 1
    n = n + 1

    if self.opt.grad == 'ac' then
      criticTurn = self.iter < self.opt.critic_warmup or self.iter % self.opt.critic_iter ~= 0
    end
  end
end

function Trainer:copyGPU(sample)
  if self.opt.nGPU > 0 then
    sample.input = sample.input:cuda()
    sample.target = sample.target:cuda()
    sample.offset = sample.offset:cuda()

    if self.opt.learnControl then
      sample.labels = sample.labels:cuda()
    end

    if self.opt.predictAngles then sample.angles = sample.angles:cuda() end
    if self.opt.predictFg then sample.fg = sample.fg:cuda() end
  end
end

function Trainer:updateCritic()
  self.optimMethodCritic(self.fevalCritic, self.criticParamsX, self.optimState.Critic)
  self.model.Critic:zeroGradParameters()
end

function Trainer:updateActor()
  local norm = self.opt.batch_size * self.opt.iter_size * self.opt.seq_length
  
  self.actorGradX:mul(1.0 / norm)
  
  if self.opt.gradClip > 0 then
    self.actorGradX:clamp(-self.opt.gradClip, self.opt.gradClip)
  end

  -- encoder
  self.optimMethod(self.fevalActor, self.actorParamsX, self.optimState.Actor)
  self.model.Actor:zeroGradParameters()
  
  -- control
  if self.opt.learnControl then
    self.controlGradX:mul(1.0 / norm)
    self.optimMethod(self.fevalControl, self.controlParamsX, self.optimState.Control)
    self.model.Control:zeroGradParameters()
  end
  
  -- decoder
  if self.opt.learnDecoder then
    if self.opt.gradClip > 0 then
      self.decoderGradX:clamp(-self.opt.gradClip, self.opt.gradClip)
    end
    self.optimMethod(self.fevalDecoder, self.decoderParamsX, self.optimState.Decoder)
    self.model.Decoder:zeroGradParameters()
  end
end

function Trainer:resetStat(info)
    info.sumReward = 0
    info.sumLossTD = 0
    info.sumLossBCE = 0
    info.sumIoU = 0
    info.sumAP = 0
    info.sumDice = 0
    info.sumCritic = 0
    info.sumKLD = 0
    info.sumMean = 0
    info.sumVar = 0
    info.sumLossCls = 0

    info.numLossTD = 0
    info.numLossBCE = 0
    info.numIoU = 0
    info.numAP = 0
    info.numDice = 0
    info.numCritic = 0
    info.numKLD = 0
    info.numMean = 0
    info.numVar = 0
    info.numReward = 0
    info.numLossCls = 0
end

function Trainer:updateInfo(info, time)
  if info.numLossTD > 0 then table.insert(self.summaryTDLoss, info.sumLossTD / info.numLossTD) end
  if info.numLossBCE > 0 then table.insert(self.summaryBCELoss, info.sumLossBCE / info.numLossBCE) end
  if info.numLossCls > 0 then table.insert(self.summaryClsLoss, info.sumLossCls / info.numLossCls) end
  if info.numKLD > 0 then table.insert(self.summaryKLLoss, info.sumKLD    / info.numKLD) end
  if info.numMean > 0 then table.insert(self.summaryKLMean, info.sumMean   / info.numMean) end
  if info.numVar > 0 then table.insert(self.summaryKLVar, info.sumVar    / info.numVar) end
  if info.numCritic > 0 then table.insert(self.summaryPLoss, info.sumCritic / info.numCritic) end
  if info.numIoU > 0 then table.insert(self.summaryIoU, info.sumIoU    / info.numIoU) end
  if info.numAP > 0 then table.insert(self.summaryAP, info.sumAP     / info.numAP) end
  if info.numDice > 0 then table.insert(self.summaryDice, info.sumDice   / info.numDice) end
  if info.numReward > 0 then table.insert(self.summaryReward, info.sumReward   / info.numReward) end
  table.insert(self.summaryTime, 1000*(sys.clock() - time))
end

function Trainer:resetInfo()
  self.summaryTime = {}
  self.summaryTDLoss = {}
  self.summaryBCELoss = {}
  self.summaryClsLoss = {}
  self.summaryKLLoss = {}
  self.summaryKLMean = {}
  self.summaryKLVar  = {}
  self.summaryPLoss = {}
  self.summaryIoU = {}
  self.summaryAP = {}
  self.summaryDice = {}
  self.summaryReward = {}
end

function Trainer:test(dataloader, tag, conf, dump, gen, stat, idd)

  local dump = dump or false
  local idd = idd or 0

  self.model.Actor:evaluate()
  self.model.Decoder:evaluate()

  if self.opt.learnControl then
    self.model.Control:evaluate()
  end

  local output_dir = string.format('val_%03d', tag)
  local save_path = paths.concat(self.opt.save, output_dir)

  if dump then
    if not paths.dirp(save_path) and not paths.mkdir(save_path) then
       cmd:error('error: unable to create checkpoint directory: ' .. save_path .. '\n')
    end
  end

  local n = 0
  local bs = self.opt.batch_size
  for sample in dataloader:run(self.opt.loadSeqLengthTest, self.cxtUpdateDataloader) do
    local time = sys.clock()
    local n_bs = sample.angles:size(1)

    if dump then
      -- saving the input
      for b = 1,n_bs do
        local nn = bs*n + b

        if self.opt.predictAngles then
          local filename = paths.concat(save_path, string.format('%02d_%02d_00_angles.png', idd, nn))
          local angle_rgb = build_orientation_img(sample.angles[b])
          if self.opt.vhigh and self.opt.predictFg then
            local fg = sample.fg[b]:float()
            angle_rgb:cmul(fg:view(1, angle_rgb:size(2), angle_rgb:size(3)):expandAs(angle_rgb))
          end
          image.save(filename, angle_rgb)
        end

        if self.opt.predictFg then
          local filename = paths.concat(save_path, string.format('%02d_%02d_00_fg.png', idd, nn))
          image.save(filename, sample.fg[1])
        end

        -- Saving the input and the ground-truth
        local in_filename = paths.concat(save_path, string.format('%02d_%02d_00_input.png', idd, nn))
        local tg_filename = paths.concat(save_path, string.format('%02d_%02d_00_target.png', idd, nn))

        local inputNorm = dataloader.dataset:renormalise(sample.input[b])
        image.save(in_filename, inputNorm)

        -- converting the GT segmentation
        local target = sample.target[b]:float()
        colormap:reset()
        local target_rgb = colormap:convert(torch.div(target, target:max()))
        image.save(tg_filename, target_rgb)

        -- external mask
        local ms_filename = paths.concat(save_path, string.format('%02d_%02d_00_extmem.png', idd, nn))
        if self.opt.contextType == 'max' then
          image.save(ms_filename, sample.extMem[b][1])
        elseif self.opt.contextType == 'angles' then
          local extMemRGB = flow.xy2rgb(sample.extMem[b][1], sample.extMem[b][2])
          image.save(ms_filename, extMemRGB)
        end
      end
    end

    -- Copy input and target to the GPU
    if self.opt.nGPU > 0 then
      sample.input = sample.input:cuda()
      sample.target = sample.target:cuda()

      if self.opt.predictAngles then sample.angles = sample.angles:cuda() end
      if self.opt.predictFg then sample.fg = sample.fg:cuda() end
    end

    -- inference
    local results = {}
    results.masks = {}
    results.extmem = {}
    results.ctrl = {}
    results.grad = {}

    if self.opt.criticGT then
      self:prepareCriticIn(sample.target)
    end

    self:stepActorMarkov(info, sample, false, results, 0, self.opt.max_seq_length)

    local masks = self:evaluate(sample, results, conf, stat)

    if dump then
      for b = 1,#masks do
        local nn = bs*n + b
        local mask = masks[b]:float()
        local mask_f = torch.div(mask, mask:max())
        local mask_rgb = colormap:convert(mask_f)
        local out_filename = paths.concat(save_path, string.format('%02d_%02d_00_output.png', idd, nn))
        image.save(out_filename, mask_rgb)

        local inputNorm = dataloader.dataset:renormalise(sample.input[b]:float())
        maskTensor = vis.getMaskTensor(mask:int())
        local rgbMask = vis.drawMasks(inputNorm, maskTensor, nil, 0.6)
        local out_filename_im = paths.concat(save_path, string.format('%02d_%02d_00_output_im.png', idd, nn))
        image.save(out_filename_im, rgbMask)
      end
      --
      for t = 1,#results.masks do
        for b = 1,results.masks[t]:size(1) do
          local nn = bs*n + b
          local mask = results.masks[t][b][1]
          local filename_m = paths.concat(save_path, string.format('%02d_%02d_%02d_mask.png', idd, nn, t))
          image.save(filename_m, mask)
          local maskCxt = results.extmem[t][b]
          if self.opt.contextType == 'angles' then
            local maskCxtRGB = flow.xy2rgb(maskCxt[1], maskCxt[2])
            local filename_e = paths.concat(save_path, string.format('%02d_%02d_%02d_extmem.png', idd, nn, t))
            image.save(filename_e, maskCxtRGB)
          else
            local filename_e = paths.concat(save_path, string.format('%02d_%02d_%02d_extmem.png', idd, nn, t))
            image.save(filename_e, maskCxt[1])
          end
        end
      end
    end

    n = n + 1
  end
end

function Trainer:evaluate(sample, result, cutoff, stat)

  -- TODO: max-matching
  -- BCE, label accuracy

  local cutoff = cutoff or 0.5
  local masks = {}
  local batchSize = sample.input:size(1)
  for b = 1,batchSize do
    local nInstances = math.min(self.opt.max_seq_length, sample.target[b]:max())

    if self.opt.learnControl then
      local nPredictions = self.opt.max_seq_length
      local labels = {}
      for t = 1,#result.ctrl do
        local labelConf, label = torch.max(result.ctrl[t][b], 1)
        if label[1] == 1 and labelConf[1] > 0.5 then
          nPredictions = t - 1
          break
        end
      end
      table.insert(stat.dic, nPredictions - nInstances)
      nInstances = nPredictions
    end

    local valid = false
    if nInstances > 0 then
      local h, w = sample.target:size(3), sample.target:size(4)
      local predMasks = torch.FloatTensor(nInstances, h, w)
      for t = 1,nInstances do
        predMasks[t]:copy(result.masks[t][b])
      end

      local predMaskMaxVal, predMaskLabel = torch.max(predMasks, 1)
      predMaskLabel[predMaskMaxVal:lt(cutoff)] = 0
      predMaskLabel = predMaskLabel[1]:int()

      if predMaskLabel:max() > 0 then
        valid = true
        table.insert(masks, predMaskLabel)

        local target = sample.target[b][1]:int()

        -- 
        -- symmetric best dice
        --
        table.insert(stat.dices, analysis.sbd(predMaskLabel, target))

        --
        -- coverage
        --
        local uwtCov, wtCov = analysis.cov(predMaskLabel, target)
        table.insert(stat.wtCov, wtCov)
        table.insert(stat.uwtCov, uwtCov)

        --
        -- Max-matching
        --
        local nInstancesGT = target:max()
        local scoreMtx = torch.DoubleTensor(nInstancesGT, nInstancesGT):zero()
        local segSize = torch.zeros(math.max(nInstancesGT, nInstances))
        local maxSegSize = 0
        for nn = 1,nInstancesGT do
          local maskGT = sample.target[b]:eq(nn):float()
          segSize[nn] = maskGT:sum()
          maxSegSize = math.max(segSize[nn], maxSegSize)
          for n = 1,math.min(nInstances, nInstancesGT) do
            local mask = predMasks:sub(n, n)
            scoreMtx[n][nn] = self:softDice(mask, maskGT)
          end
        end
        local ok, assignments = pcall(hungarian.maxCost, scoreMtx)
        local _, sortSize = torch.sort(segSize, 1, true)
        local _, segRank = torch.sort(sortSize, 1)

        --
        -- stepwise
        --
        for t = 1,math.min(nInstances, nInstancesGT) do
          local targetIdx = assignments[t]
          local targetMask = sample.target[b]:eq(targetIdx):float()
          local predMask = predMaskLabel:eq(t):float()
          --
          local iou = analysis.iou(predMask, targetMask, cutoff)
          local dice = analysis.dice(predMask, targetMask, cutoff)
          --
          stat.iou[t] = stat.iou[t] + iou
          stat.dice[t] = stat.dice[t] + dice
          stat.segSize[t] = stat.segSize[t] + segSize[targetIdx] / maxSegSize
          stat.counts[t] = stat.counts[t] + 1
          --
          local gtRank = segRank[targetIdx]
          stat.sizeCorr[t][gtRank] = stat.sizeCorr[t][gtRank] + 1
        end
      end
    end

    if not valid then
      table.insert(stat.dices, 0)
      table.insert(stat.wtCov, 0)
      table.insert(stat.uwtCov, 0)
    end
  end

  return masks
end


function Trainer:saveModels(tag)
  self:clearStates()

  -- Saving the models
  io.write('-- Saving the models...')
  time = sys.clock()

  self.model.Actor:clearState()
  torch.save(paths.concat(self.opt.save, string.format('%03d_actor.model', tag)), self.model.Actor)
  torch.save(paths.concat(self.opt.save, string.format('%03d_actor_opt.t7', tag)), self.optimState.Actor)

  if self.opt.learnControl then
    self.model.Control:clearState()
    torch.save(paths.concat(self.opt.save, string.format('%03d_control.model', tag)), self.model.Control)
    torch.save(paths.concat(self.opt.save, string.format('%03d_control_opt.t7', tag)), self.optimState.Control)
  end

  if self.opt.learnDecoder then
    self.model.Decoder:clearState()
    torch.save(paths.concat(self.opt.save, string.format('%03d_decoder.model', tag)), self.model.Decoder)
    torch.save(paths.concat(self.opt.save, string.format('%03d_decoder_opt.t7', tag)), self.optimState.Decoder)
  end

  if self.opt.grad == 'ac' then
    self.model.Critic:clearState()
    torch.save(paths.concat(self.opt.save, string.format('%03d_critic.model', tag)),  self.model.Critic)
    torch.save(paths.concat(self.opt.save, string.format('%03d_critic_opt.t7', tag)), self.optimState.Critic)
  end

  io.write(' Done (' .. 1000*(sys.clock() - time) .. 'ms)\n')
end

function Trainer:bufferIsFull()
  return self.batchIdx == self.opt.iter_size*self.opt.batch_size
end

function Trainer:resetBuffer()
  self.batchIdx = 0
  for t = 1,self.opt.max_seq_length do 
    self.batchData.data[t].batchSize = 0
    self.batchData.data[t].index = {} -- index in the global batch
  end
  self.batchData.length = 0

  collectgarbage()
  collectgarbage()
end

function Trainer:stepCritic(info, sample)

  local batchSize = sample.target:size(1)
  local maxLabels, _ = torch.max(sample.target:view(batchSize, -1), 2)
  maxLabels = maxLabels:long()

  local nTimesteps = maxLabels:max()
  local debugStr = ''

  for t = 1,nTimesteps do
    local bIndices = maxLabels:ge(t):nonzero()[{{}, 1}]
 
    --
    -- critic prediction
    --
    local criticIn = {sample.input:index(1, bIndices),
                      self.batchData.data[t].extMem:index(1, bIndices):cuda(),
                      self.batchData.data[t].actionMask:index(1, bIndices):cuda()}

    if self.opt.criticGT then
       table.insert(criticIn, self.batchData.targetAngles:index(1, bIndices))
    else
       if self.opt.predictAngles then table.insert(criticIn, sample.angles:index(1, bIndices)) end
       if self.opt.predictFg then table.insert(criticIn, sample.fg:index(1, bIndices)) end
    end

    if self.opt.criticTime then
      local offsets = sample.offset:index(1, bIndices)
      local timestep = 1.0 - (t + offsets) / self.opt.max_seq_length
      table.insert(criticIn, timestep)
    end
 
    local criticOut = self.model.Critic:forward(criticIn)
    local criticTarget = self.batchData.data[t].rewards:clone()

    local gamma = self.opt.gamma
    for tt = t + 1,nTimesteps do
      local ttIndices = maxLabels:ge(tt):nonzero()[{{}, 1}]
      local rewards = self.batchData.data[tt].rewards:index(1, ttIndices)
      criticTarget:indexAdd(1, ttIndices, gamma * rewards)
      gamma = gamma * gamma
    end

    criticTarget = criticTarget:index(1, bIndices):cuda()

    local lossTD = self.criterion.TD:forward(criticOut, criticTarget)
    local gradTD = self.criterion.TD:backward(criticOut, criticTarget)

    self.model.Critic:backward(criticIn, gradTD)

    if bIndices[1] == 1 then
      local r = self.batchData.data[t].rewards[1]:squeeze()
      local pred = criticOut[1]:squeeze()
      local tgt = criticTarget[1]:squeeze()
      local grad = gradTD[1]:squeeze()
      local err = math.abs(pred - tgt) / (1e-3 + criticTarget[1]:abs():squeeze())
      debugStr = debugStr .. string.format('\n= [%02d] %5.4f [%5.4f] (GT %5.4f) | Err %5.4f | Grad %5.4f', t, pred, r, tgt, err, grad) 
    end

    info.sumLossTD = info.sumLossTD + (criticOut - criticTarget):abs():sum()
    info.numLossTD = info.numLossTD + bIndices:nElement()
  end
  print(debugStr)
end

function Trainer:stepActor(info, sample)

  local nPixels = self.opt.imHeight * self.opt.imWidth

  -- starting from the initial state
  local rnnStatesNext = {}
  for i = 1,#self.init_state_batch do
    table.insert(rnnStatesNext, self.init_state_batch[i]:zero())
  end

  local cRnnStatesPrev
  if self.opt.grad == 'ac' then
    if self.opt.criticVer == 'lstm' then
      cRnnStatesPrev = {self.criticInit:clone(), self.criticInit:clone()}
    elseif self.opt.criticVer == 'irnn' or self.opt.criticVer == 'irnn_pool' then
      cRnnStatesPrev = self.criticInit:clone()
    end
  end

  local actionGrads = {}
  local ctrlGrads = {}

  local batchSize = sample.input:size(1)
  self.extMem:sub(1, batchSize):copy(sample.extMem)
  self.maskSum:sub(1, batchSize):copy(sample.maskSum)

  local actInputs = {}
  local decInputs = {}

  --
  -- Assembling the input for the actor
  --
  local cnnIn = self.batchData.cnnIn
  local extMem = self.extMem
  local maskSum = self.maskSum
  local angleMask = self.batchData.angleMask
  local fgMask = self.batchData.fg

  -- collecting the gradient from the critic
  -- to the mask and RNN states
  for t = 1,self.opt.seq_length do

    actInputs[t] = {}
    for tt = 1,#rnnStatesNext do
      table.insert(actInputs[t], rnnStatesNext[tt]:clone())
    end

    table.insert(actInputs[t], cnnIn)

    local extMemActor = extMem:clone()
    if torch.uniform() < self.opt.maskNoise then extMemActor:fill(1) end
    table.insert(actInputs[t], extMemActor)

    local decInput = {0, cnnIn}

    if self.opt.predictAngles then
      table.insert(actInputs[t], angleMask)
      if self.opt.pyramidAngles then table.insert(decInput, angleMask) end
    end

    if self.opt.predictFg then
      table.insert(actInputs[t], fgMask)
      if self.opt.pyramidFg then table.insert(decInput, fgMask) end
    end

    local actOut = self.modelActorStates[t]:forward(actInputs[t])

    if not self.opt.blockHidden then
      for tt = 1,#rnnStatesNext do
        rnnStatesNext[tt]:copy(actOut[1 + tt])
      end
    end

    --
    -- Learning control
    --
    if self.opt.learnControl then
      local ctrlIn = rnnStatesNext[#rnnStatesNext]
      local ctrlOut = self.model.Control:forward(ctrlIn)
      --
      local ctrlTarget = self.batchData.data[t].targetLabels
      --
      -- Class Loss & Gradient
      --
      local lossCE = self.criterion.CE:forward(ctrlOut, ctrlTarget)
      local gradCE = self.criterion.CE:backward(ctrlOut, ctrlTarget)

      local ctrlGrad = self.model.Control:backward(ctrlIn, gradCE)
      info.sumLossCls = info.sumLossCls + lossCE
      info.numLossCls = info.numLossCls + self.opt.batch_size

      table.insert(ctrlGrads, ctrlGrad:clone())
    end

    -- Decoding the mask
    decInput[1] = actOut[1]:clone() -- copying the code/action

    local masks
    if self.opt.learnDecoder then
      decInputs[t] = {}
      for ii = 1,#decInput do table.insert(decInputs[t], decInput[ii]) end
      masks = self.modelDecoderStates[t]:forward(decInputs[t])
    else
      masks = self.model.Decoder:forward(decInput)
    end

    -- Computing the stats
    local bIndices = self.batchData.data[t].batchIndices

    if bIndices and bIndices:nElement() > 0 then
      local targetMaskF = self.batchData.data[t].targetMask:index(1, bIndices)

      local mask = masks:index(1, bIndices)
      local maskF = mask:float()

      local batchIoU  = self:softIoU(maskF, targetMaskF)
      local batchDice = self:softDice(maskF, targetMaskF)

      info.sumIoU  = info.sumIoU  + batchIoU:sum()
      info.numIoU  = info.numIoU  + bIndices:nElement()

      info.sumDice = info.sumDice + batchDice:sum()
      info.numDice = info.numDice + bIndices:nElement()

      info.sumMean = info.sumMean + self.modelActorStates[t].KLDLoss.avgMean
      info.numMean = info.numMean + bIndices:nElement()

      info.sumVar = info.sumVar  + self.modelActorStates[t].KLDLoss.avgVar
      info.numVar = info.numVar + bIndices:nElement()

      info.sumKLD = info.sumKLD  + self.modelActorStates[t].KLDLoss.loss
      info.numKLD = info.numKLD + bIndices:nElement()

      local targetMask = targetMaskF:cuda()

      --
      -- policy gradient
      --
      local decoderGrad = self.maskZero:zero()
      if self.opt.grad == 'fixed' then
        local bceLoss = self.criterion.BCE:forward(mask, targetMask) / nPixels
        decoderGrad:indexCopy(1, bIndices, self.opt.betaBCE * self.criterion.BCE:backward(mask, targetMask) / nPixels)
        info.sumLossBCE = info.sumLossBCE + bceLoss
        info.numLossBCE = info.numLossBCE + bIndices:nElement()
      else -- grad == 'ac'
        local mask_ = mask:index(1, bIndices)
        local cnnIn_ = cnnIn:index(1, bIndices)
        local extMem_ = extMem:index(1, bIndices)
        local angleMask_ = angleMask:index(1, bIndices)
        local fgMask_ = fgMask:index(1, bIndices)

        --
        -- critic prediction
        --
        local nActive = bIndices:nElement()
        local criticIn = {}

        if self.opt.criticVer == 'lstm' then
          for j = 1,#cRnnStatesPrev do
            table.insert(criticIn, cRnnStatesPrev[j]:index(1, bIndices))
          end
          table.insert(criticIn, cnnIn_)
          table.insert(criticIn, mask_)
          table.insert(criticIn, extMem_)
        elseif self.opt.criticVer == 'irnn' or self.opt.criticVer == 'irnn_pool' then
          criticIn = {cnnIn_, cRnnStatesPrev, mask_, extMem_}
        else
          criticIn = {cnnIn_, extMem_, mask_}
        end

        if self.opt.predictAngles then table.insert(criticIn, angleMask_) end
        if self.opt.predictFg then table.insert(criticIn, fgMask_) end

        local criticOut = self.model.Critic:forward(criticIn)

        if self.opt.criticVer == 'lstm' then
          local gradCritic = {}
          for j = 1,#self.gradCritic do
            table.insert(gradCritic, self.gradCritic[j]:sub(1, nActive))
          end

          for j = 1,#cRnnStatesPrev do
            cRnnStatesPrev[j]:indexCopy(1, bIndices, criticOut[j + 1])
          end

          decoderGrad = self.model.Critic:updateGradInput(criticIn, gradCritic)[4]
          info.sumCritic = info.sumCritic + criticOut[1]:sum()
        elseif self.opt.criticVer == 'irnn' or self.opt.criticVer =='irnn_pool' then
          local gradCritic = {}
          for j = 1,#self.gradCritic do
            table.insert(gradCritic, self.gradCritic[j]:sub(1, nActive))
          end
          decoderGrad = self.model.Critic:updateGradInput(criticIn, gradCritic)[3]
          cRnnStatesPrev:indexCopy(1, bIndices, criticOut[2])
          info.sumCritic = info.sumCritic + criticOut[1]:sum()
        else
          local gradCritic = self.gradCritic:sub(1, nActive)
          decoderGrad = self.model.Critic:updateGradInput(criticIn, gradCritic)[3]
          info.sumCritic = info.sumCritic + criticOut:sum()
        end

        info.numCritic = info.numCritic + bIndices:nElement()

        if self.opt.withBCELoss then
          local bceLoss = self.criterion.BCE:forward(mask, targetMask)
          local bceGrad = self.criterion.BCE:backward(mask, targetMask)
          decoderGrad:indexAdd(1, bIndices, self.opt.betaBCE * bceGrad)
          info.sumLossBCE = info.sumLossBCE + bceLoss
          info.numLossBCE = info.numLossBCE + bIndices:nElement()
        end
      end

      local actionGrad
      if self.opt.learnDecoder then
        actionGrad = self.modelDecoderStates[t]:backward(decInputs[t], decoderGrad)[1]
      else
        actionGrad = self.model.Decoder:updateGradInput(decInput, decoderGrad)[1]
      end
  
      self.actionZero:copy(actionGrad)
    end
    --
    table.insert(actionGrads, self.actionZero:clone())
    self.actionZero:zero()
    --
		self.cxtUpdate(extMem, masks, maskSum)
  end
  -- 
  -- Actor gradient
  --
  for tt = 1,#self.rnnGrad do self.rnnGrad[tt]:zero() end
  --
  --
  for t = self.opt.seq_length,1,-1 do
    --
    local actionGrad = {actionGrads[t], unpack(self.rnnGrad)}
    --
    if self.opt.learnControl then
      -- CTRL gradient through LSTM state
      actionGrad[#actionGrad]:add(ctrlGrads[t])
    end
    --
    -- built-in CTRL in the actor
    if not self.opt.oldControl then
      table.insert(actionGrad, self.clsGrad)
    end
    --
    local gradActor = self.modelActorStates[t]:backward(actInputs[t], actionGrad)
    --
    if t > 1 then
      if not self.opt.blockHidden then
        for tt = 1,#self.rnnGrad do
          self.rnnGrad[tt]:copy(gradActor[tt])
        end
      end
    end
  end
end


function Trainer:stepActorMarkov(info, sample, train, res, noise, len)

  local seqLength = len or self.opt.seq_length
  local batchSize = sample.input:size(1)
  local nPixels = self.opt.imHeight * self.opt.imWidth

  -- starting from the initial state
  local rnnStatesNext = {}
  for i = 1,#self.init_state_batch do
    local rnnState = self.init_state_batch[i]:zero()
    table.insert(rnnStatesNext, rnnState:sub(1, batchSize))
  end

  self.extMem:sub(1, batchSize):copy(sample.extMem)
  self.maskSum:sub(1, batchSize):copy(sample.maskSum)

  local cnnIn = sample.input
  local extMem = self.extMem:sub(1, batchSize)
  local maskSum = self.maskSum:sub(1, batchSize)
  local angleMask = sample.angles
  local fgMask = sample.fg

  local maxLabel, _ = torch.max(sample.target:view(batchSize, -1), 2)
  maxLabel = maxLabel:long()

  local nTimesteps = math.min(seqLength, maxLabel:max())

  if self.opt.learnControl then
    nTimesteps = self.opt.max_seq_length
  end

  -- Insert random noise
  -- at a randomly sampled timestep
  local noiseT = 0
  if noise > math.random() then
    noiseT = math.random(1, nTimesteps)
  end

  -- 
  -- actor input
  --
  local actInput = {}

  for tt = 1,#rnnStatesNext do
    table.insert(actInput, rnnStatesNext[tt])
  end

  table.insert(actInput, cnnIn)
  table.insert(actInput, extMem)

  --
  -- decoder input
  --
  local decInput = {0}

  if self.opt.pyramidImage then table.insert(decInput, cnnIn) end

  if self.opt.predictAngles then
    table.insert(actInput, angleMask)
    if self.opt.pyramidAngles then table.insert(decInput, angleMask) end
  end

  if self.opt.predictFg then
    table.insert(actInput, fgMask)
    if self.opt.pyramidFg then table.insert(decInput, fgMask) end
  end

  --
  -- critic input
  --
  local criticIn = {cnnIn, extMem, 0}

  if self.opt.criticGT then
    local bIndices = maxLabel:ge(1):nonzero()[{{}, 1}]
    local targetAngles = self.batchData.targetAngles:index(1, bIndices)
    table.insert(criticIn, targetAngles)
  else
    if self.opt.predictAngles then table.insert(criticIn, angleMask) end
    if self.opt.predictFg then table.insert(criticIn, fgMask) end
  end

  local criticTime
  if self.opt.criticTime then
    criticTime = torch.zeros(maxLabel:size()):cuda()
    table.insert(criticIn, criticTime)
  end

  for t = 1,nTimesteps do

    if t == noiseT then
      -- turning on noise for this iteration
      self:noSampling(false)
    end

    self.model.Actor:setTimestep(t)
    local actOut = self.model.Actor:forward(actInput)
    decInput[1] = actOut[1] -- getting the code/action

    if not self.opt.blockHidden then
      for tt = 1,#rnnStatesNext do
        rnnStatesNext[tt]:copy(actOut[1 + tt])
      end
    end

    --
    -- Learning control
    --
    local ctrlGrad
    if self.opt.learnControl then
      --
      local ctrlIn = rnnStatesNext[#rnnStatesNext]
      local ctrlOut = self.model.Control:forward(ctrlIn)
      --
      -- Class Loss & Gradient
      --
      if train then
        local ctrlTarget = sample.labels:sub(1, -1, t, t) + 1
        local lossCE = self.criterion.CE:forward(ctrlOut, ctrlTarget)
        local gradCE = self.criterion.CE:backward(ctrlOut, ctrlTarget)
        ctrlGrad = self.model.Control:backward(ctrlIn, gradCE)
        --
        info.sumLossCls = info.sumLossCls + lossCE
        info.numLossCls = info.numLossCls + self.opt.batch_size
      else
        -- just accumulate the result
        table.insert(res.ctrl, ctrlOut:float())
      end
    end

    -- Decoding the mask
    local masks = self.model.Decoder:forward(decInput)
    local bIndices = maxLabel:ge(t):nonzero()

    --
    -- Backward pass
    --
    if train then
      self.actionZero:zero()
      if bIndices:nElement() > 0 then
        bIndices = bIndices[{{}, 1}]

        local mask = masks:index(1, bIndices)
        local maskF = mask:float()

        local targetIndex = self.targetAssignment:sub(1, -1, t, t):index(1, bIndices)
        local gtMasks = sample.target:index(1, bIndices):typeAs(targetIndex)

        targetIndex = targetIndex:view(-1, 1, 1, 1):expandAs(gtMasks)

        local targetMask = targetIndex:eq(gtMasks):typeAs(mask)
        local targetMaskF = targetMask:typeAs(maskF)

        local batchIoU  = self:softIoU(maskF, targetMaskF)
        local batchDice = self:softDice(maskF, targetMaskF)

        info.sumIoU  = info.sumIoU  + batchIoU:sum()
        info.numIoU  = info.numIoU  + bIndices:nElement()

        info.sumDice = info.sumDice + batchDice:sum()
        info.numDice = info.numDice + bIndices:nElement()

        info.sumMean = info.sumMean + self.model.Actor.KLDLoss.avgMean
        info.numMean = info.numMean + bIndices:nElement()

        info.sumVar = info.sumVar  + self.model.Actor.KLDLoss.avgVar
        info.numVar = info.numVar + bIndices:nElement()

        info.sumKLD = info.sumKLD  + self.model.Actor.KLDLoss.loss
        info.numKLD = info.numKLD + bIndices:nElement()

        local fElements = bIndices:nElement() / self.opt.batch_size

        --
        -- policy gradient
        --
        local decoderGrad = self.maskZero:zero()
        if self.opt.grad == 'fixed' then
          local bceLoss = self.criterion.BCE:forward(mask, targetMask)
          local bceGrad = self.criterion.BCE:backward(mask, targetMask)
          local bceGradNorm = self.opt.betaBCE * bceGrad * fElements

          decoderGrad:indexCopy(1, bIndices, bceGradNorm)
          info.sumLossBCE = info.sumLossBCE + bceLoss --/ nPixels
          info.numLossBCE = info.numLossBCE + bIndices:nElement()
        else -- if grad == 'ac' then
          --
          -- critic prediction
          --
          criticIn[3] = masks
          if self.opt.criticTime then
            criticTime[{}] = 1.0 - t / nTimesteps
          end
          --
          local criticOut = self.model.Critic:forward(criticIn)
          local criticGrad = self.model.Critic:updateGradInput(criticIn, self.gradCritic)
          local criticGradSub = fElements * criticGrad[3]:index(1, bIndices)
          decoderGrad:indexCopy(1, bIndices, criticGradSub)
          --
          info.sumCritic = info.sumCritic + criticOut:sum()
          info.numCritic = info.numCritic + bIndices:nElement()
        end
        --
        local actionGrad
        if self.opt.learnDecoder then
          actionGrad = self.model.Decoder:backward(decInput, decoderGrad)[1]
        else
          actionGrad = self.model.Decoder:updateGradInput(decInput, decoderGrad)[1]
        end
        --
        self.actionZero:indexCopy(1, bIndices, actionGrad:index(1, bIndices))
      end
      --
      -- Backward pass
      --
      local actionGrads = {self.actionZero}
      --
      for tt = 1,#self.rnnGrad do
        table.insert(actionGrads, self.rnnGrad[tt]:zero())
      end
      --
      if self.opt.learnControl and ctrlGrad then
        actionGrads[#actionGrads]:add(ctrlGrad)
      end
      --
      -- control gradient
      --
      if not self.opt.oldControl then
        table.insert(actionGrads, self.clsGrad)
      end
      --
      self.model.Actor:backward(actInput, actionGrads)
    else
      if self.opt.grad == 'ac' then
        criticIn[3] = masks
        if self.opt.criticTime then
          criticTime:copy(t * maxLabel:float():pow(-1))
        end
        --
        local bs = masks:size(1)
        local gradCritic = self.gradCritic:sub(1, bs)

        --
        local criticOut = self.model.Critic:forward(criticIn)
        local criticGrad = self.model.Critic:updateGradInput(criticIn, gradCritic)
        table.insert(res.grad, criticGrad[3]:float())
      end
      -- adding the predicted masks
      table.insert(res.masks, masks:float())
      table.insert(res.extmem, extMem:float())
    end
    --
    self.cxtUpdate(extMem, masks, maskSum)
    --
    if t == noiseT then
      self:noSampling(true)
    end
  end
end

function Trainer:prepareCriticIn(target)
  local batchSize = target:size(1)
  local maxLabels, _ = torch.max(target:view(batchSize, -1), 2)
  maxLabels = maxLabels:long()

  local nInstances = maxLabels:max()
  self.batchData.targetAngles:zero()
  for t = 1,nInstances do
    -- selecting sub-batch with GT left
    local bIndices = maxLabels:ge(t):nonzero()[{{}, 1}]
    local masks = target:index(1, bIndices):eq(t):typeAs(target)
    local maskAngles = self.getMaskAngles:forward(masks)
    self.batchData.targetAngles:indexAdd(1, bIndices, maskAngles:cuda())
  end
end

function Trainer:playBatch(info, sample, explore, seqLength)
  local seqLength = seqLength or self.opt.seq_length
  local batchSize = sample.target:size(1)
  local maxLabels, _ = torch.max(sample.target:view(batchSize, -1), 2)
  maxLabels = maxLabels:long()

  -- starting from the initial state
  local rnnStatesNext = {}
  for i = 1,#self.init_state_batch do
    local rnnState = self.init_state_batch[i]:zero()
    table.insert(rnnStatesNext, rnnState:sub(1, batchSize))
  end

  local currentReward = torch.zeros(batchSize, 1)

  self:noSampling(explore < math.random())

  local nInstances = maxLabels:max()
  local nTimesteps = math.min(seqLength, nInstances)

  if self.opt.criticGT then
    self:prepareCriticIn(sample.target)
  end

  -- Assignment matrix
  local scoreMat = torch.DoubleTensor(batchSize, nTimesteps, nInstances):zero()
  self.targetAssignment = torch.LongTensor(self.opt.batch_size, nTimesteps)

  -- preparing the masks
  self.extMem:sub(1, batchSize):copy(sample.extMem)
  self.maskSum:sub(1, batchSize):copy(sample.maskSum)

  for t = 1,nTimesteps do
    -- selecting sub-batch with GT left
    local bIndices = maxLabels:ge(t):nonzero()[{{}, 1}]

    local cnnIn = sample.input:index(1, bIndices)
    local extMem = self.extMem:index(1, bIndices)
    local maskSum = self.maskSum:index(1, bIndices)

    -- updating the state for the critic
    self.batchData.data[t].extMem:indexCopy(1, bIndices, extMem:float())
    self.batchData.data[t].maskSum:indexCopy(1, bIndices, maskSum:float())

    local actInput = {}
    for tt = 1,#rnnStatesNext do
      table.insert(actInput, rnnStatesNext[tt]:index(1, bIndices))
    end

    table.insert(actInput, cnnIn)
    table.insert(actInput, extMem)

    local decInput = {0}

    if self.opt.pyramidImage then table.insert(decInput, cnnIn) end

    local angleMask
    if self.opt.predictAngles then
      angleMask = sample.angles:index(1, bIndices)
      table.insert(actInput, angleMask)
      if self.opt.pyramidAngles then table.insert(decInput, angleMask) end
    end

    local fgMask
    if self.opt.predictFg then
      fgMask = sample.fg:index(1, bIndices)
      table.insert(actInput, fgMask)
      if self.opt.pyramidFg then table.insert(decInput, fgMask) end
    end

    self.model.Actor:setTimestep(t)
    local actOut = self.model.Actor:forward(actInput)

    -- passing the action to the decoder
    decInput[1] = actOut[1] 

    -- copying the next LSTM state
    if not self.opt.blockHidden then
      for tt = 1,#rnnStatesNext do
        rnnStatesNext[tt]:indexCopy(1, bIndices, actOut[1 + tt])
      end
    end

    -- predicting the mask
    local mask = self.model.Decoder:forward(decInput)

    if t == 1 and self.opt.realNoise > math.random() then
      -- substituting with best-matching GT mask
      local targetSub = sample.target:index(1, bIndices)
      local bestGt = self.bestScoreIndex:forward({mask, targetSub})
      local bestGtIndex = bestGt:sub(1, -1, 1, 1):typeAs(targetSub)
      bestGtIndex = bestGtIndex:view(-1, 1, 1, 1):expandAs(targetSub)
      mask:copy(bestGtIndex:eq(targetSub):typeAs(mask))
    end

    if self.opt.ignoreMem then
      mask[torch.gt(extMem, mask)] = 0
    end

    -- augmenting the batch
    self.batchData.data[t].actionMask:indexCopy(1, bIndices, mask:float())

    -- Computing the scores for assignment
    local time = sys.clock()
    --
    local scoreMatch = scoreMat:index(1, bIndices)
    local targetMatch = sample.target:index(1, bIndices)
    local matchResult = self.batchMatchIndex:forward({scoreMatch:sub(1, -1, 1, t), targetMatch, mask})
    scoreMat:indexCopy(1, bIndices, scoreMatch)
    --
    local nextReward, assignment = unpack(matchResult)
    assignment = assignment:long()
    --
    local reward = nextReward - currentReward:index(1, bIndices)
    self.batchData.data[t].rewards:indexCopy(1, bIndices, reward)
    --
    if self.opt.logReward then
      if self.opt.normScore then
        local reward = reward:cmul(reward):cdiv(1e-5 + scoreMatch:sub(1, -1, t, t):sum(3):float())
        self.batchData.data[t].rewards:indexCopy(1, bIndices, reward)
      end
      self.batchData.data[t].rewards:add(1e-5):log()
    elseif self.opt.logRewardNorm then
      self.batchData.data[t].rewards:div(self.opt.logRewardNormB):add(1):log()
    end
    --
    -- reward stat
    currentReward:indexCopy(1, bIndices, nextReward)
    info.sumReward = info.sumReward + self.batchData.data[t].rewards:index(1, bIndices):sum()
    info.numReward = info.numReward + bIndices:nElement()
    --
    self.targetAssignment:sub(1, -1, 1, t):indexCopy(1, bIndices, assignment:sub(1, -1, 1, t))
    --
    self.cxtUpdate(extMem, mask, maskSum)
    --
    self.extMem:indexCopy(1, bIndices, extMem)
    self.maskSum:indexCopy(1, bIndices, maskSum)
    --
    -- turning off exploration (cheap)
    self:noSampling(true)
  end
end

function Trainer:allocateBatch()
  self.batchData = {}
  self.batchData.data = {}
  self.batchData.cnnIn = torch.zeros(self.opt.batch_size, self.opt.imageCh, self.opt.imHeight, self.opt.imWidth)
  self.batchData.angleMask = torch.zeros(self.opt.batch_size, self.opt.numAngles, self.opt.imHeight, self.opt.imWidth)
  self.batchData.fg = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
  self.batchData.binFg = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth) -- binarised FG
  self.batchData.nInstances = {}
  self.batchData.nTimesteps = {}

  if self.opt.criticGT then
    self.batchData.targetAngles = torch.zeros(self.opt.batch_size, 2, self.opt.imHeight, self.opt.imWidth)
  end

  if self.opt.nGPU > 0 then
    self.batchData.cnnIn = self.batchData.cnnIn:cuda()
    self.batchData.angleMask = self.batchData.angleMask:cuda()
    self.batchData.fg = self.batchData.fg:cuda()
    self.batchData.binFg = self.batchData.binFg:cuda()
    if self.opt.criticGT then
      self.batchData.targetAngles = self.batchData.targetAngles:cuda()
    end
  end

  for t = 1,self.opt.max_seq_length do
    self.batchData.data[t] = {}
    self.batchData.data[t].extMem = torch.zeros(self.opt.batch_size, self.opt.cxtSize, self.opt.imHeight, self.opt.imWidth)
    self.batchData.data[t].maskSum = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
    self.batchData.data[t].rewards = torch.zeros(self.opt.batch_size, 1)
    self.batchData.data[t].targetMask = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
    self.batchData.data[t].targetLabels = torch.zeros(self.opt.batch_size)
    self.batchData.data[t].actionMask = torch.zeros(self.opt.batch_size, 1, self.opt.imHeight, self.opt.imWidth)
    self.batchData.data[t].ctrl = torch.zeros(self.opt.batch_size, 1)
    self.batchData.data[t].rnnState = {}
  end
end

function Trainer:noSampling(on)
  self.model.Actor.sampler.data.module.fixed = on
end

function Trainer:selectAction(decInput, criticIn, nsamples)
  local actor = self.model.Actor
  local critic = self.model.Critic
  local decoder = self.model.Decoder

  nsamples = math.max(nsamples, 1)

  function getAction(z)
    local out = z
    for i = 75,79 do
      out = actor.forwardnodes[i].data.module:forward(out)
    end
    return out
  end

  -- getting the predicted mean and the variance
  local sampler = actor.sampler
  local kldiv = actor.kldiv
  local meanvar = kldiv.data.module.output

  -- baseline: no sampling
  sampler.data.module.fixed = true
  sampler.data.module.train = false

  local bestScore = -1e5
  local bestAction = nil
  local baselineScore = -1

  for n = 1,nsamples do
    local act = sampler.data.module:forward(meanvar)
    local decAct = getAction(act)

    -- decoder
    decInput[1] = decAct
    masks = decoder:forward(decInput)


    -- critic
    criticIn[3] = masks
    local criticOut = self.model.Critic:forward(criticIn)

    -- enabling sampling
    sampler.data.module.fixed = false
    sampler.data.module.train = true

    if criticOut[1][1] > bestScore or n == 1 then
      bestScore = criticOut[1][1]
      bestAction = decAct:clone()
      if n == 1 then
        baselineScore = criticOut[1][1]
      end
    end
  end

  print('Baseline = ' .. baselineScore .. ' / bestScore = ' .. bestScore)

  return bestAction
end

function Trainer:clearStates()
  -- clearning the states (to save memory for validation)

  self.model.Decoder:clearState()
  if self.opt.learnDecoder and self.modelDecoderStates then
    for t = 1,#self.modelDecoderStates do
      self.modelDecoderStates[t]:clearState()
    end
  end

  if self.opt.grad == 'ac' then
    self.model.Critic:clearState()
    if self.opt.criticVer == 'lstm' or self.opt.criticVer == 'irnn' or self.opt.criticVer == 'irnn_pool' then
      for t = 1,#self.modelCriticStates do
        self.modelCriticStates[t]:clearState()
      end
    end
  end

  self.model.Actor:clearState()
  for t = 1,#self.modelActorStates do
    self.modelActorStates[t]:clearState()
  end

  if self.opt.learnControl then
    self.model.Control:clearState()
  end

  collectgarbage()
  collectgarbage()
end

function Trainer:softDice(a, b)
  local batchSize = a:size(1)
  assert(batchSize == b:size(1), 'Batch size mismatch')

  local a = a:float():view(batchSize, -1)
  local b = b:float():view(batchSize, -1)

  local ab = torch.sum(torch.cmul(a, b), 2)
  local aa = torch.sum(a, 2)
  local bb = torch.sum(b, 2)
  return torch.cdiv(2*ab, aa + bb)
end


function Trainer:softIoU(a, b)
  local batchSize = a:size(1)
  assert(batchSize == b:size(1), 'Batch size mismatch')

  local a = a:float():view(batchSize, -1)
  local b = b:float():view(batchSize, -1)

  local ab = torch.sum(torch.cmul(a, b), 2)
  local aa = torch.sum(a, 2)
  local bb = torch.sum(b, 2)
  return torch.cdiv(ab, aa + bb - ab)
end

function Trainer:dice(pred, truth, conf)
  local batchSize = pred:size(1)
  assert(batchSize == truth:size(1), 'Batch size mismatch')

  local conf = conf or 0.5

  -- [0, 1] => {0, 1}
  local p = torch.ge(pred, conf):float():view(batchSize, -1)
  local t = truth:float():view(batchSize, -1)

  local pSum = torch.sum(p, 2)
  local tSum = torch.sum(t, 2)
  local ptDot = torch.sum(torch.cmul(p, t), 2)
  local batchDice = torch.cdiv(ptDot, pSum + tSum + 1e-12)

  return 2 * batchDice
end

function Trainer:AP(pred, truth, avgs, threshold)
  local avgs = avgs or self.avgs
  local batchSize = pred:size(1)
  assert(batchSize == truth:size(1), 'Batch size mismatch')

  local pred = pred:float()
  local truth = truth:float()

  local threshold = threshold or 0.5

  -- computing the confidence as average non-zero value
  local nonZero = torch.gt(pred, 0):float()
  local nonZeroSum = torch.sum(nonZero:view(batchSize, -1), 2)
  local maskSum = torch.sum(pred:view(batchSize, -1), 2)
  local confVal = torch.cdiv(maskSum, nonZeroSum)
  local _, confIdx = torch.sort(confVal, 1)
  local confMat = torch.zeros(batchSize, batchSize)
  for i = 1,batchSize do
    confMat[i][confIdx[i][1]] = 1
  end

  -- [0, 1] => {0, 1}
  local predHard = torch.ge(pred, threshold):float()

  local pSum = torch.sum(predHard:view(batchSize, -1), 2)
  local tSum = torch.sum(truth:view(batchSize, -1), 2)

  -- precision
  local tp = torch.sum(torch.cmul(predHard:view(batchSize, -1), truth:view(batchSize, -1)), 2)
  local fp = pSum - tp
  local pr = torch.cdiv(tp, (tp + fp):add(1e-12))

  -- IoU
  local pt = pSum + tSum - tp
  local iou = torch.cdiv(tp, pt)

  -- sorting the IoU by confidence
  local iouSorted = torch.mm(confMat, iou)
  iouSorted = torch.repeatTensor(iouSorted, 1, avgs:size(1))

  local avgs = torch.repeatTensor(avgs, batchSize, 1)
  local prs = torch.repeatTensor(pr, 1, avgs:size(2)):cmul(torch.ge(iouSorted, avgs):float())

  -- ordering the IoUs by the confidence
  local precCum = torch.cumsum(prs, 2)
  local denom = torch.repeatTensor(torch.range(1, avgs:size(2)), batchSize, 1)

  return precCum:cdiv(denom):mean(2)
end


function Trainer:getRGBMask(seqMask, conf)
  local conf = conf or 0
  local nInstances = seqMask:size(1)
  local gt_mask = torch.zeros(seqMask:size(2), seqMask:size(3))
  for i = 1, nInstances do
    local mask_nz = torch.eq(gt_mask, 0):float()
    local mask_i = torch.cmul(mask_nz, torch.gt(seqMask[i], conf):float())
    gt_mask:add(torch.mul(mask_i, i):div(nInstances))
  end
  local bkg_mask = torch.gt(gt_mask, 0):double()
  local gt_mask_rgb = colormap:convert(gt_mask)
  return torch.cmul(gt_mask_rgb, bkg_mask:repeatTensor(3, 1, 1))
end

return M.Trainer
