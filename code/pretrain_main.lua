require 'common'
require 'os'

local models = require 'models/init'
local DataLoader = require 'dataloader'
local PreTrainer = require 'pretrain'
local opts = require 'opts'

require 'colormap'
colormap:setStyle('parula')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
opt.mode = 'pretrain'

-- create log file
inGPUMode = opt.nGPU > 1
if inGPUMode == 1 then
  cutorch.setDevice(opt.gpu_id)
end

if opt.manualSeed ~= 0 then
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)
  math.randomseed(opt.manualSeed)
else
  torch.seed()
  cutorch.seed()
  math.randomseed(os.time())
end

-- default lambdaKL for training is 0.01
--opt.lambdaKL = 1000

--nngraph.setDebug(true)

-- if opt.preproc == '' then
--   print('PreprocNet is not specified!')
-- end

local model = {}

-- Loading the preproc net
-- local preprocVer, preprocEpoch = unpack(string.split(opt.preproc, ','))
-- local preprocPrefix = string.format(opt.save_dir .. '/%s_%s/%03d_', opt.dataset, preprocVer, preprocEpoch)
-- print('Loading ' .. preprocPrefix .. 'preproc.model')
-- model.preproc = torch.load(preprocPrefix .. 'preproc.model')

local fromEpoch = 1
local optimState
if opt.continue ~= '' then
  local ver
  ver, fromEpoch = unpack(string.split(opt.continue, ','))
  fromEpoch = tonumber(fromEpoch)

  local snapPrefix = string.format(opt.save_dir .. '/%s_%s/%03d_', opt.dataset, ver, fromEpoch)
  optimState = {}

  -- loading encoder
  if not opt.noEnc then
    print('Loading ' .. snapPrefix .. 'cvae_enc.t7')
    model.enc = torch.load(snapPrefix .. 'cvae_enc.t7')

    if model.enc.sampler then
      print('Found sampler: sampler.fixed = ', model.enc.sampler.data.module.fixed)
      model.enc.sampler.data.module.fixed = opt.noSampling
      print('Set sampler: sampler.fixed = ', model.enc.sampler.data.module.fixed)
    end
    
    print('Loading ' .. snapPrefix .. 'cvae_opt_enc.t7')
    optimState.Enc = torch.load(snapPrefix .. 'cvae_opt_enc.t7')

    model.KLDLoss = model.enc.kldiv.data.module

    print('-- lambdaKL = ' .. model.KLDLoss.KLDK)
    if model.KLDLoss.KLDK ~= opt.lambdaKL then
      model.KLDLoss.KLDK = opt.lambdaKL
      print('-- Setting lambdaKL = ' .. model.KLDLoss.KLDK)
    end
  end

  -- loading decoder
  print('Loading ' .. snapPrefix .. 'cvae_dec.t7')
  model.dec = torch.load(snapPrefix .. 'cvae_dec.t7')

  print('Loading ' .. snapPrefix .. 'cvae_opt_dec.t7')
  optimState.Dec = torch.load(snapPrefix .. 'cvae_opt_dec.t7')
else

  if not opt.noEnc then
    model.enc = models.setup('cvae_encoder', opt, checkpoint)

    if model.enc.sampler then
      print('Found sampler: sampler.fixed = ', model.enc.sampler.data.module.fixed)
      model.enc.sampler.data.module.fixed = opt.noSampling
      print('Set sampler: sampler.fixed = ', model.enc.sampler.data.module.fixed)
    end
  
    model.KLDLoss = model.enc.kldiv.data.module
  end

  model.dec = models.setup('cvae_decoder', opt, checkpoint)
end


local criterion = {}
if opt.semSeg then
  criterion.SEG = cudnn.SpatialCrossEntropyCriterion()
else
  criterion.SEG = nn.BCECriterion(nil, false)
end

if opt.nGPU > 0 then criterion.SEG = criterion.SEG:cuda() end

-- Data loading
local trainLoader = DataLoader.create(opt, 'train')
local valLoader = DataLoader.create(opt, 'val')

-- The trainer handles the training loop and evaluation on validation set
local trainer = PreTrainer(model, criterion, opt, optimState)

if opt.checkOnly then
  --validate(opt, 'train', fromEpoch - 1, false, trainLoader)
  trainer:test(valLoader, fromEpoch, 0.5, true)
  os.exit(1)
end

print('Starting pre-training from epoch ' .. fromEpoch)
for epoch = fromEpoch + 1, opt.nEpochs do
  -- Train for a single epoch
  trainer:train(epoch, trainLoader)

  if epoch % opt.checkpoint_after == 0 then
    trainer:saveModels(epoch)
    trainer:test(valLoader, epoch, 0.5, true)
  elseif epoch % opt.validation_after == 0 then
    print("Train:\n")
    trainer:test(trainLoader, epoch, 0.5, false)
    print("Val:\n")
    trainer:test(valLoader, epoch, 0.5, false)
  end
end
