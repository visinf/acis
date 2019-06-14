require 'common'
require 'os'

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


function getRGBMask(seqMask, conf)
  local mask = seqMask:float()
  mask:div(mask:max())
  local bg_mask = mask:gt(0):float()
  local gt_mask_rgb = colormap:convert(mask):float()
  return torch.cmul(gt_mask_rgb, bg_mask:repeatTensor(3, 1, 1))
end

-- Create model
local fromEpoch = 1

local model = {}
local optimState = {}
local hyperparams = {}

if opt.continue ~= '' then
  local ver
  ver, fromEpoch = unpack(string.split(opt.continue, ','))
  fromEpoch = tonumber(fromEpoch)

  local snapPrefix = string.format('checkpoints/%s_%s/%03d_', opt.dataset, ver, fromEpoch)

  print('Loading model from ' .. snapPrefix .. '*')
  model.preproc = torch.load(snapPrefix .. 'preproc.model')

  print('Loading Optimisation State')
  optimState = {}
  optimState.preproc = torch.load(snapPrefix .. 'preproc_opt.t7')

  fromEpoch = fromEpoch + 1
else
  model.preproc = models.setup('preproc', opt, checkpoint)

  optimState.preproc = {learningRate = opt.learning_rate,
                        learningRateDecay = opt.learning_rate_decay,
                        weightDecay = opt.weightDecay}
end

-- Data loading
local trainLoader = DataLoader.create(opt, 'train')
local valLoader = DataLoader.create(opt, 'val')

local gen = torch.Generator()
torch.manualSeed(gen, opt.manualSeed)

-- allocating the batch
local imageIn = torch.Tensor(opt.batch_size, opt.imageCh, opt.imHeight, opt.imWidth)

local nOutChannels = 0
if opt.predictAngles then nOutChannels = nOutChannels + opt.numAngles end
if opt.predictFg then nOutChannels = nOutChannels + 1 end

if nOutChannels == 0 then
  print("Nothing to learn. Set -predictAngles or -predictFg")
  os.exit(1)
end

local angleMask = torch.IntTensor(opt.batch_size, opt.imHeight, opt.imWidth)
local fgMask = torch.Tensor(opt.batch_size, 1, opt.imHeight, opt.imWidth)

local angleMaskX = torch.IntTensor(1, opt.imHeight, opt.imWidth)
local fgMaskX = torch.Tensor(1, 1, opt.imHeight, opt.imWidth)

if opt.nGPU > 0 then
  imageIn = imageIn:cuda()
  fgMask = fgMask:cuda()
  angleMask = angleMask:cuda()
  fgMaskX = fgMaskX:cuda()
  angleMaskX = angleMaskX:cuda()
end

local spSoftMax = cudnn.SpatialSoftMax()
if opt.nGPU > 0 then
  spSoftMax = spSoftMax:cuda()
end

-- initialising criterion
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

local preprocParams, preprocGrad = model.preproc:getParameters()

feval = function()
  return info.angleLoss + info.fgLoss, preprocGrad
end

optimMethod = optim.adam

model.preproc:zeroGradParameters()
local timer = torch.Timer()

print('Starting training...')
local iter = 1
for epoch = fromEpoch, opt.nEpochs do

  model.preproc:training()

  for sample in trainLoader:run() do
    ----------------- Preparing the batch --------------

    imageIn:copy(sample.input)

    if opt.predictAngles then
      _, a_mask = sample.angles:max(2)
      angleMask:copy(a_mask)
    end

    if opt.predictFg then
      fgMask:copy(sample.fg)
    end

    local pred = model.preproc:forward(imageIn)
  
    local grad = {}
    if opt.predictAngles then
      info.angleLoss = info.angleLoss + criterion.angles:forward(pred[1], angleMask)
      table.insert(grad, criterion.angles:backward(pred[1], angleMask))
    end
  
    if opt.predictFg then
      info.fgLoss = info.fgLoss + criterion.fg:forward(pred[2], fgMask)
      table.insert(grad, criterion.fg:backward(pred[2], fgMask))
    end
  
    model.preproc:backward({imageIn}, grad)
  
    if iter % opt.iter_size == 0 then
      optimMethod(feval, preprocParams, optimState.preproc)
      model.preproc:zeroGradParameters()
    end

    info.count = info.count + 1
    iter = iter + 1

    if iter % opt.summary_after == 0 then
      io.write(string.format('%08.1f [%06d/%03d] ', timer:time().real, iter, epoch))
      io.write(string.format(' | AngleLoss %4.3e / FGLoss %4.3e\n', info.angleLoss / info.count, info.fgLoss / info.count))
      io.flush()
  
      info.angleLoss = 0
      info.fgLoss = 0
      info.count = 0

      collectgarbage()
      collectgarbage()
    end
  end
  
  -- Validation? Checkpoint?
  local angleLoss = 0
  local fgLoss = 0
  local count = 0

  if epoch % opt.checkpoint_after == 0 or
       epoch % opt.validation_after == 0 then
  
    model.preproc:evaluate()

    local dump = (epoch % opt.checkpoint_after == 0)

    local output_dir = string.format('val_%03d', epoch)
    local save_path = paths.concat(opt.save, output_dir)

    if dump then
      model.preproc:clearState()

      collectgarbage()
      collectgarbage()

      -- saving the model
      torch.save(paths.concat(opt.save, string.format('%03d_preproc.model', epoch)), model.preproc)
      torch.save(paths.concat(opt.save, string.format('%03d_preproc_opt.t7', epoch)), optimState.preproc)

      if not paths.dirp(save_path) and not paths.mkdir(save_path) then
         cmd:error('error: unable to create checkpoint directory: ' .. save_path .. '\n')
      end
    end
  
    local n = 1
    for sample in valLoader:run() do

      local sampleIn = sample.input
      if opt.nGPU > 0 then
        sampleIn = sampleIn:cuda()
      end

      if dump then
        -- Saving the input and the ground-truth
        local in_filename = paths.concat(save_path, string.format('%02d_01_input.png', n))
        local tg_filename = paths.concat(save_path, string.format('%02d_02_target.png', n))

        local inputNorm = valLoader.dataset:renormalise(sample.input[1])
        image.save(in_filename, inputNorm)

        -- converting the GT segmentation
        local rgbMask = getRGBMask(sample.target[1])
        image.save(tg_filename, rgbMask)
      end
  
      if opt.predictAngles then
        local y, i = torch.max(sample.angles, 2)
        angleMaskX[1]:copy(i[1])
      end
  
      if opt.predictFg then
        fgMaskX[1]:copy(sample.fg[1])
      end
  
      -- batch is complete: learn!
      local pred = model.preproc:forward(sampleIn)
  
      if opt.predictAngles then
        angleLoss = angleLoss + criterion.angles:forward(pred[1]:sub(1, 1), angleMaskX)
        local angles = spSoftMax:forward(pred[1])
        if dump then
          local angle_rgb = build_orientation_img(angles[1]:float())
          local filename = paths.concat(save_path, string.format('%02d_04_angles.png', n))
          image.save(filename, angle_rgb)
        end
      end
  
      if opt.predictFg then
        fgLoss = fgLoss + criterion.fg:forward(pred[2]:sub(1, 1), fgMaskX[1])
        if dump then
          local filename_m = paths.concat(save_path, string.format('%02d_03_fg.png', n))
          image.save(filename_m, pred[2][1])
        end
      end

      count = count + 1
      n = n + 1
    end

    io.write(string.format('Validation: \n'))
    io.write(string.format('AngleLoss: %4.3e\n', angleLoss / count))
    io.write(string.format('   FGLoss: %4.3e\n', fgLoss / count))
    io.flush()
  end
end
