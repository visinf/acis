require 'common'
require 'os'
require 'utils/instance_map'

--
-- Preprocessing the complete dataset and saving the result
-- once and for all
--

local models = require 'models/init'
local DataLoader = require 'dataloader'
local Trainer = require 'train'
local opts = require 'opts'

require 'utils/ContextManager'
require 'colormap'
colormap:setStyle('parula')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

opt.mode = 'preproc'
opt.maxEpochSize = 1e6

if opt.manualSeed ~= 0 then
  print('Setting the seed ' .. opt.manualSeed)
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)
  math.randomseed(opt.manualSeed)
else
  torch.seed()
  cutorch.seed()
  math.randomseed(os.time())
end


-- Create model
local model = {}

if opt.continue ~= '' then
  local ver
  ver, fromEpoch = unpack(string.split(opt.continue, ','))
  fromEpoch = tonumber(fromEpoch)

  local snapPrefix = string.format('checkpoints/%s_%s/%03d_', opt.dataset, ver, fromEpoch)

  print('Loading model from ' .. snapPrefix .. '*')
  model.preproc = torch.load(snapPrefix .. 'preproc.model')
  model.preproc:cuda()

  if opt.cudnn == 'deterministic' then
    print('Preprocessing: deterministic')
    model.preproc:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end
else
  print('No preprocessing model specified')
  os.exit(1)
end

-- Data loading
local trainLoader = DataLoader.create(opt, 'train')
local valLoader = DataLoader.create(opt, 'val')

local gen = torch.Generator()
torch.manualSeed(gen, opt.manualSeed)

local nOutChannels = 0
if opt.predictAngles then nOutChannels = nOutChannels + opt.numAngles end
if opt.predictFg then nOutChannels = nOutChannels + 1 end

if nOutChannels == 0 then
  print("Nothing to learn. Set -predictAngles or -predictFg")
  os.exit(1)
end

-- allocating the batch
local imageIn = torch.Tensor(opt.batch_size, opt.imageCh, opt.imHeight, opt.imWidth)
local angleMask = torch.IntTensor(opt.batch_size, opt.imHeight, opt.imWidth)
local fgMask = torch.Tensor(opt.batch_size, 1, opt.imHeight, opt.imWidth)

if opt.nGPU > 0 then
  imageIn = imageIn:cuda()
  fgMask = fgMask:cuda()
  angleMask = angleMask:cuda()
end

local spSoftMax = cudnn.SpatialSoftMax()
if opt.nGPU > 0 then spSoftMax = spSoftMax:cuda() end

-- criterion
local criterion = {}
if opt.predictAngles then
  criterion.angles = cudnn.SpatialCrossEntropyCriterion()
  if opt.nGPU > 0 then
    criterion.angles = criterion.angles:cuda()
  end
end
if opt.predictFg then
  criterion.fg = nn.BCECriterion()
  if opt.nGPU > 0 then
    criterion.fg = criterion.fg:cuda()
  end
end

-- Train for a single epoch
local info = {}
info.angleLoss = 0
info.fgLoss = 0
info.count = 0

local timer = torch.Timer()
--local loaders = {val = valLoader}
--local loaders = {val = valLoader}
local loaders = {train = trainLoader, val = valLoader}
local numRuns = opt.preproc_epoch

model.preproc:training()
for tag,loader in pairs(loaders) do
  print('Starting preprocessing [' .. tag .. ']')

  local sample_idx = 0
  for ii = 1,numRuns do
    xlua.progress(ii, numRuns)
    for sample in loader:run() do
      imageIn:copy(sample.input)

      if opt.predictAngles then 
        _, a_mask = sample.angles:max(2)
        angleMask:copy(a_mask)
      end
      if opt.predictFg then fgMask:copy(sample.fg) end

      local pred = model.preproc:forward(imageIn)

      if opt.predictAngles then
        info.angleLoss = info.angleLoss + criterion.angles:forward(pred[1], angleMask)
        pred[1] = spSoftMax:forward(pred[1]):float()
      end

      if opt.predictFg then
        info.fgLoss = info.fgLoss + criterion.fg:forward(pred[2], fgMask)
        pred[2] = pred[2]:float()
      end

      loader.dataset:saveData(sample_idx, pred, sample)
      info.count = info.count + 1
      sample_idx = sample_idx + 1
    end
  end

  io.write(string.format('%08.1f [%s] ', timer:time().real, tag))
  io.write(string.format(' | AngleLoss %4.3e / FGLoss %4.3e\n', info.angleLoss / info.count, info.fgLoss / info.count))
  io.flush()
  
  info.angleLoss = 0
  info.fgLoss = 0
  info.count = 0
end
