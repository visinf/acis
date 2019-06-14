local analysis = require 'analysis'
local image = require 'image'

require 'utils/instance_map'
require 'utils/ContextManager'
require 'modules/SamplerVAE'
require 'modules/BestScoreIndex'


local M = {}
local PreTrainer = torch.class('ris.PreTrainer', M)

function PreTrainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.criterion = criterion
  self.optimState = optimState

  if not self.optimState then
    print("Initialising optimState")
    self.optimState = {}
    self.optimState.Enc = {learningRate = opt.learning_rate, weightDecay = opt.weightDecay}
    self.optimState.Dec = {learningRate = opt.learning_rate, weightDecay = opt.weightDecay}
  else
    print("Updating optimState")
    self.optimState.Enc.learningRate = opt.learning_rate
    self.optimState.Enc.weightDecay = opt.weightDecay

    self.optimState.Dec.learningRate = opt.learning_rate
    self.optimState.Dec.weightDecay = opt.weightDecay
  end

  print('optimState.Enc = ', self.optimState.Enc)
  print('optimState.Dec = ', self.optimState.Dec)
  self.spSoftMax = cudnn.SpatialSoftMax()
  if opt.nGPU > 0 then self.spSoftMax = self.spSoftMax:cuda() end

  self.opt = opt
  --self.optimMethod = optim.sgd
  self.optimMethod = optim.adam
  --self.optimMethod = optim.rmsprop

  self.model.enc:zeroGradParameters()
  self.model.dec:zeroGradParameters()
  self.encParams, self.encGrad = self.model.enc:getParameters()
  self.decParams, self.decGrad = self.model.dec:getParameters()

  print('Encoder has > ' .. self.encParams:nElement() .. ' < parameters')
  print('Decoder has > ' .. self.decParams:nElement() .. ' < parameters')
  
  self.summaryLoss = torch.Tensor(opt.summary_after):fill(0)
  self.summaryKLErr = torch.Tensor(opt.summary_after):fill(0)
  self.summaryMean = torch.Tensor(opt.summary_after):fill(0)
  self.summaryVar = torch.Tensor(opt.summary_after):fill(0)
  self.summaryTime = torch.Tensor(opt.summary_after):fill(0)
  self.summaryIoU = torch.Tensor(opt.summary_after):fill(0)
  self.summaryAP = torch.Tensor(opt.summary_after):fill(0)
  self.summaryDice = torch.Tensor(opt.summary_after):fill(0)

  -- preparing the IoU thresholds
  self.avgs = torch.Tensor({0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95})

  local cxtManagerNonBatch = ContextManager(opt, true)
  if opt.contextType == 'max' then
    -- applying pixelwise max with the memory and the mask
    -- dataloader uses the same 
    self.cxtUpdate = function(x,y,z) cxtManagerNonBatch:max(x,y,z)  end
  elseif opt.contextType == 'angles' then
    -- merging the mask in 2-dimensional angle representation (x,y)
	  self.cxtUpdate = function(x,y,z) cxtManagerNonBatch:angles(x,y,z) end
  end

  -- eval function
  self.fevalEnc = function() return -1, self.encGrad end
  self.fevalDec = function() return -1, self.decGrad end

  self.timer = torch.Timer()

  -- helper modules
  self.bestScoreIndex = nn.BestScoreIndex()

  self.iter = 1
end

function PreTrainer:train(epoch, dataloader)

  self.model.enc:training()
  self.model.dec:training()

  -- segmenting the next instance
  local n = 0
  for sample in dataloader:run(self.opt.seq_length, self.cxtUpdate) do

    local time = sys.clock()
    local info = self:trainStep(sample, self.iter)

    local x_index = self.iter % opt.summary_after + 1
    self.summaryLoss[x_index] = info.sumLoss / info.sumCount
    self.summaryKLErr[x_index] = info.sumKLErr / info.sumCount
    self.summaryMean[x_index] = info.sumMean / info.sumCount
    self.summaryVar[x_index] = info.sumVar / info.sumCount
    self.summaryAP[x_index]   = info.sumAP / info.sumCount
    self.summaryTime[x_index] = 1000*(sys.clock() - time)
    self.summaryIoU[x_index]  = info.sumIoU / info.sumCount
    self.summaryDice[x_index] = info.sumDice / info.sumCount

    if self.iter % self.opt.summary_after == 0 then
      io.write(string.format('%08.1f [%04d/%04d] ', self.timer:time().real, epoch, n))
      io.write(string.format('%5.3f ms | RLoss %4.3f | KL Err %4.3e | Mean %4.3f | Var %4.3e | IoU %4.3f | AP %4.3f | Dice %4.3f\n', 
                                  torch.mean(self.summaryTime),
                                  torch.mean(self.summaryLoss),
                                  torch.mean(self.summaryKLErr),
                                  torch.mean(self.summaryMean),
                                  torch.mean(self.summaryVar),
                                  torch.mean(self.summaryIoU),
                                  torch.mean(self.summaryAP),
                                  torch.mean(self.summaryDice)))
      io.flush()
    end

    self.iter = self.iter + 1
    n = n + 1

    collectgarbage()
    collectgarbage()
  end

end

function PreTrainer:getInputDataTest(sample, label, bIndices)
  local subSample = {}
  subSample.input = sample.input:index(1, bIndices)
  subSample.target = sample.target:index(1, bIndices)

  if self.opt.pyramidAngles or self.opt.predictAngles then
      subSample.angles = sample.angles:index(1, bIndices)
  end

  if self.opt.predictFg then
      subSample.fg = sample.fg:index(1, bIndices)
  end

  if self.opt.nGPU > 0 then
    subSample.input = subSample.input:cuda()
    subSample.target = subSample.target:cuda()
  end

  local fg = torch.gt(subSample.target, 0):typeAs(subSample.input)
  local targetMask = torch.eq(subSample.target, label):typeAs(subSample.input)

  local encInput = {subSample.input, targetMask}
  local decInput = {0}

  self:augmentData(encInput, decInput, subSample)

  return encInput, decInput, targetMask
end

function PreTrainer:getInputData(sample)
  if self.opt.nGPU > 0 then
    sample.input = sample.input:cuda()
    sample.target = sample.target:cuda()
    sample.extMem = sample.extMem:cuda()
    sample.maskSum = sample.maskSum:cuda()
  end

  local encInput = {sample.input, sample.target}
  local decInput = {0}

  self:augmentData(encInput, decInput, sample)

  return encInput, decInput
end

function PreTrainer:augmentData(encInput, decInput, sample)

  if self.opt.pyramidImage then
    table.insert(decInput, sample.input)
  end

  if self.opt.predictAngles then
    if self.opt.nGPU > 0 then sample.angles = sample.angles:cuda() end

    table.insert(encInput, sample.angles)

    if self.opt.pyramidAngles then
      table.insert(decInput, sample.angles)
    end
  end

  if self.opt.predictFg then
    if self.opt.nGPU > 0 then sample.fg = sample.fg:cuda() end

    table.insert(encInput, sample.fg)

    if self.opt.pyramidFg then
      table.insert(decInput, sample.fg)
    end
  end

  return encInput, decInput
end

function PreTrainer:trainStep(sample, iter)

  local info = {}
  info.sumLoss = 0
  info.sumAP = 0
  info.sumIoU = 0
  info.sumDice = 0
  info.sumCount = 0
  info.sumKLErr = 0
  info.sumMean = 0
  info.sumVar = 0
  info.sumCount = 0

  local encInput, decInput = self:getInputData(sample)

  local encOut = self.model.enc:forward(encInput)
  decInput[1] = encOut

  local mask = self.model.dec:forward(decInput)

  -- finding the max IoU target
  local bestMask = self.bestScoreIndex:forward({mask, sample.target})
  
  local bestMaskIndex = bestMask:sub(1, -1, 1, 1):typeAs(sample.target)
  bestMaskIndex = bestMaskIndex:view(-1, 1, 1, 1):expandAs(sample.target)
  local targetMask = bestMaskIndex:eq(sample.target):typeAs(mask)

  local loss = self.criterion.SEG:forward(mask, targetMask)
  local gradMask = self.criterion.SEG:backward(mask, targetMask)
  local gradAct = self.model.dec:backward(decInput, gradMask)

  self.model.enc:backward(encInput, gradAct[1])

  info.sumAP = info.sumAP + self:AP(mask, targetMask, self.avgs, 0.5):mean()
  info.sumDice = info.sumDice + self:softDice(mask, targetMask):mean()
  info.sumIoU  = info.sumIoU + self:softIoU(mask, targetMask):mean()
  info.sumLoss = info.sumLoss + loss / self.opt.batch_size
  info.sumKLErr = info.sumKLErr + self.model.KLDLoss.loss / self.opt.batch_size
  info.sumMean = info.sumMean + self.model.KLDLoss.avgMean / self.opt.batch_size
  info.sumVar = info.sumVar + self.model.KLDLoss.avgVar / self.opt.batch_size
  info.sumCount = info.sumCount + 1

  if iter % self.opt.iter_size == 0 then
    if self.opt.iter_size > 1 then
      self.encGrad:mul(1.0 / self.opt.iter_size)
      self.decGrad:mul(1.0 / self.opt.iter_size)
    end

    if self.opt.gradClip > 0 then
      self.encGrad:clamp(-self.opt.gradClip, self.opt.gradClip)
      self.decGrad:clamp(-self.opt.gradClip, self.opt.gradClip)
    end

    self.optimMethod(self.fevalEnc, self.encParams, self.optimState.Enc)
    self.optimMethod(self.fevalDec, self.decParams, self.optimState.Dec)
    self.model.enc:zeroGradParameters()
    self.model.dec:zeroGradParameters()
  end

  return info
end

function PreTrainer:test(dataloader, tag, conf, save)
  local save = save or false

  self.model.enc:evaluate()
  self.model.dec:evaluate()

  local output_dir = string.format('val_%03d', tag)
  local save_path = paths.concat(self.opt.save, output_dir)

  if save and not paths.dirp(save_path) and not paths.mkdir(save_path) then
     cmd:error('error: unable to create checkpoint directory: ' .. save_path .. '\n')
  end

  --
  -- TODO: checking on last 1 instance
  --
  local sumLoss = 0
  local sumIoU = 0
  local sumAP = 0
  local sumDice = 0
  local globalDice = 0
  local globalCount = 0
  local niter = 0
  local count = 0

  for sample in dataloader:run(self.opt.max_seq_length, self.cxtUpdate) do
    local time = sys.clock()

    local bs, c, h, w = unpack(sample.input:size():totable())
    local maxLabel, _ = torch.max(sample.target:view(bs, -1), 2)

    local totalLoss = torch.zeros(bs, 1)
    local totalIoU = torch.zeros(bs, 1)
    local totalAP = torch.zeros(bs, 1)
    local totalDice = torch.zeros(bs, 1)

    local nInstances = maxLabel:max()

    --local totalWtCov = torch.zeros(nInstances)
    --local totalUwtCov = torch.zeros(nInstances)
    --local totalMean = torch.zeros(nInstances)
    --local totalVar = torch.zeros(nInstances)

    if save then
      for b = 1,bs do
        local im = sample.input[b]:float()

        -- Saving the input and the ground-truth
        local in_filename = paths.concat(save_path, string.format('%02d_%02d_00_input.png', niter, b))
        local tg_filename = paths.concat(save_path, string.format('%02d_%02d_00_target.png', niter, b))
        local inputNorm = dataloader.dataset:renormalise(im)
        image.save(in_filename, inputNorm)

        -- converting the GT segmentation
        local target = sample.target[b]:float()
        local target_rgb = colormap:convert(torch.div(target, target:max()))
        image.save(tg_filename, target_rgb)
      end
    end

    for n = 1,nInstances do
      local bIndices = maxLabel:ge(n):nonzero()[{{}, 1}]
      local encInput, decInput, targetMask = self:getInputDataTest(sample, n, bIndices)

      ------------------- forward pass -------------------

      local encOut = self.model.enc:forward(encInput)
      decInput[1] = encOut

      local mask = self.model.dec:forward(decInput)

      if save then
        local maskIn = sample.extMem:float()
        for b = 1,bIndices:nElement() do
          local bb = bIndices[b]

          local filename_tt = paths.concat(save_path, string.format('%02d_%02d_%02d_mask_in.png', niter, bb, n))
          image.save(filename_tt, encInput[2][b])

          local filename_t = paths.concat(save_path, string.format('%02d_%02d_%02d_mask_out.png', niter, bb, n))
          image.save(filename_t, mask[b]:float())

          local filename_t = paths.concat(save_path, string.format('%02d_%02d_%02d_target.png', niter, bb, n))
          image.save(filename_t, targetMask[b]:float())

          for ii = 1,encOut:size(2) do
            local encImg = encOut[b][ii]:float()
            encImg = (encImg - encImg:min()) / (encImg:max() - encImg:min())
            encImg = image.scale(encImg, 20*encImg:size(2), 20*encImg:size(1), 'simple')
            local filename_tt = paths.concat(save_path, string.format('%02d_%02d_%02d_mask_z_%03d.png', niter, b, n, ii))
            image.save(filename_tt, encImg)
          end
        end
      end

      totalIoU:indexAdd(1, bIndices, self:softIoU(mask, targetMask))
      totalAP:indexAdd(1, bIndices, self:AP(mask, targetMask, self.avgs, 0.5))
      totalDice:indexAdd(1, bIndices, self:dice(mask, targetMask, 0.5))
    end

    globalDice = globalDice + totalDice:sum()
    globalCount = globalCount + maxLabel:sum()

    local norm = maxLabel:float()
    sumIoU = sumIoU + totalIoU:cdiv(norm):sum()
    sumAP = sumAP + totalAP:cdiv(norm):sum()
    sumDice = sumDice + totalDice:cdiv(norm):sum()
    count = count + bs

    niter = niter + 1
  end

  print('\nTest results:')
  print('--------------')
  print(string.format('  Mean IoU: %f', sumIoU / count))
  print(string.format('  Total AP: %f', sumAP / count))
  print(string.format('Total Dice: %f', sumDice / count))
  print(string.format('Global Dice: %f', globalDice / globalCount))
  print('------------------------\n')
end

function PreTrainer:softIoU(a, b)
  local batchSize = a:size(1)
  assert(batchSize == b:size(1), 'Batch size mismatch')

  local a = a:float():view(batchSize, -1)
  local b = b:float():view(batchSize, -1)

  local ab = torch.sum(torch.cmul(a, b), 2)
  local aa = torch.sum(a, 2)
  local bb = torch.sum(b, 2)
  return torch.cdiv(ab, aa + bb - ab)
end

-- weighted coverage
function PreTrainer:mCov(pred, truth, conf, weighted)
  -- pred [B, T, H, W]
  -- truth [B, T, H, W]

  local conf = conf or 0

  local batchSize = pred:size(1)
  assert(batchSize == truth:size(1), 'Batch size mismatch')

  local nInstances = truth:size(2)

  -- [0, 1] => {0, 1}
  local pr = torch.ge(pred, conf):float():view(batchSize, nInstances, -1)
  local tr = truth:float():view(batchSize, nInstances, -1)

  local prr = torch.sum(pr, 3)
  local cov = torch.Tensor(batchSize, nInstances)
  local tr_sum = tr:sum()
  for i = 1,nInstances do
    local tr_i = tr:sub(1, -1, i, i):expandAs(pr)
    local ab = torch.cmul(tr_i, pr):sum(3)
    local trr = tr_i:sum(3)
    local w = 1 / nInstances
    if weighted then w = tr:sub(1, -1, i, i):sum() / tr_sum end
    cov[{{}, i}] = w * torch.cdiv(ab, prr + trr - ab):max(2)
  end

  return cov:sum(2)
end

function PreTrainer:softDice(pred, truth)
  local batchSize = pred:size(1)
  assert(batchSize == truth:size(1), 'Batch size mismatch')

  local p = pred:float():view(batchSize, -1)
  local t = truth:float():view(batchSize, -1)

  local pSum = torch.sum(p, 2)
  local tSum = torch.sum(t, 2)
  local ptDot = torch.sum(torch.cmul(p, t), 2)
  local batchDice = torch.cdiv(ptDot, pSum + tSum + 1e-12)

  return 2 * batchDice
end

function PreTrainer:dice(pred, truth, avgs, conf)
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

function PreTrainer:AP(pred, truth, avgs, threshold)
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

function PreTrainer:saveModels(tag)
  io.write('-- Saving the model...')
  self.model.enc:clearState()
  self.model.dec:clearState()
  torch.save(paths.concat(self.opt.save, string.format('%03d_cvae_enc.t7', tag)), self.model.enc)
  torch.save(paths.concat(self.opt.save, string.format('%03d_cvae_dec.t7', tag)), self.model.dec)
  torch.save(paths.concat(self.opt.save, string.format('%03d_cvae_opt_enc.t7', tag)), self.optimState.Enc)
  torch.save(paths.concat(self.opt.save, string.format('%03d_cvae_opt_dec.t7', tag)), self.optimState.Dec)
  io.write('-- Done Saving the model...')
end

function PreTrainer:getRGBMask(seqMask, conf)
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

return M.PreTrainer
