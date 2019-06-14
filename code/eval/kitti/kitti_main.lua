--
-- Evaluation Script for CVPPP leaf dataset
--
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'image'

local cv = require 'cv'
require 'cv.imgproc'

--npy4th = require 'npy4th'

-- custom
local DataLoader = require 'kitti_data'
require 'KLDCriterion'
require 'SamplerVAE'
require 'SpatialDownSampling'

torch.setdefaulttensortype('torch.FloatTensor')

--
-- Options
--
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 ')
cmd:text()
cmd:text('Options:')

 ------------ General options --------------------
cmd:option('-dataIn',         '', 'Input directory with data')
cmd:option('-dataOut',        '', 'Output directory for results')
cmd:option('-cache',     'cache', 'Output directory for results')
cmd:option('-gpu',          true, 'Flag for using a GPU')
cmd:option('-maxLength',      16, 'Maximum number of predictions')
cmd:option('-modelIdx',      '1', 'Maximum number of predictions')
cmd:option('-enCxt',       false, 'Enable CXT')
cmd:option('-vo',          false, 'Visualise Order')
cmd:option('-vi',              0, 'Visualise interpolation between the predictions with X steps')
cmd:option('-vc',          false, 'Visualise Mask')
cmd:option('-vr',          false, 'Visualise Mask (random colours)')
cmd:option('-vraw',        false, 'Visualise Raw Masks')
cmd:option('-pps',         false, 'Post-processing')
cmd:option('-cutoff',        0.5, 'Confidence threshold for the masks')
cmd:option('-alpha',         0.6, 'Alpha Channel for visualisation')
cmd:option('-exportNpy',   false, 'Export Numpy mask')
cmd:option('-nThreads',        4, 'Number of data loading threads')

cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- constants
opt.ySize = 256
opt.xSize = 768
opt.nStates = 4

if opt.vo or opt.vc or opt.vr then
  require 'utils'
end

--
-- Post-processing structs
--
local dKernel = torch.ones(5, 5)

--
--
--
function interpolate(state_to, state_from, decIn, dec, n_steps, fg)
  local masks = {}
  local ffg = fg:float()
  for n = 1,n_steps do
    local alpha = (n - 1) / (n_steps - 1)
    local state = (1 - alpha) * state_from + alpha * state_to
    decIn[1] = state
    local msk = dec:forward(decIn):float()
    if fg then msk:cmul(ffg) end
    table.insert(masks, msk)
  end
  return masks
end

function getDirFromPath(baseDir, samplePath, suffix)
  local suffix = suffix or ''
  local dirName = string.gsub(paths.basename(samplePath), '_rgb.png', suffix)
  local dirPath = paths.concat(baseDir, dirName)
  if not paths.dirp(dirPath) and not paths.mkdir(dirPath) then
    cmd:error('Error / unable to create output directory: ' .. dirPath .. '\n')
  end
  return dirPath
end

--
--
--
function loadModel(idx)
  local withCxt = false
  local dir = string.format('models/%s', idx)

  local model = {}
  local actPath = paths.concat(dir, 'act.t7')
  assert(paths.filep(actPath), "Model act NOT found: " .. actPath)
  model.act = torch.load(actPath)
  model.act:evaluate()

  local decPath = paths.concat(dir, 'dec.t7')
  assert(paths.filep(decPath), "Model dec NOT found: " .. decPath)
  model.dec = torch.load(decPath)
  model.dec:evaluate()

  local ctlPath = paths.concat(dir, 'ctl.t7')
  assert(paths.filep(ctlPath), "Model ctl NOT found: " .. ctlPath)
  model.ctl = torch.load(ctlPath)
  model.ctl:evaluate()

  local ppcPath = paths.concat(dir, 'ppc.t7')
  assert(paths.filep(ppcPath), "Model ppc NOT found: " .. ppcPath)
  model.ppc = torch.load(ppcPath)
  model.ppc:evaluate()

  local cxtPath = paths.concat(dir, 'cxt.t7')
  if paths.filep(cxtPath) then
    withCxt = true
    model.cxt = torch.load(cxtPath)
    model.cxt:evaluate()
  end

  for i, net in ipairs(model) do
    net:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end

  return model, withCxt
end

--
-- Reading the data
--
local data = DataLoader.create(opt)
local extMem = torch.Tensor(1, 1, opt.ySize, opt.xSize)
local cxtMem = torch.Tensor(1, 3, opt.ySize, opt.xSize)
local state = torch.Tensor(1, 512)

if opt.gpu then
  extMem = extMem:cuda()
  cxtMem = cxtMem:cuda()
  state = state:cuda()
end

local states = {}
for i = 1,opt.nStates do table.insert(states, state:clone()) end

local spSoftMax = cudnn.SpatialSoftMax()
if opt.gpu then spSoftMax = spSoftMax:cuda() end

local softMax = cudnn.SpatialSoftMax()
if opt.gpu then softMax = softMax:cuda() end

---
-- Main Program
---
local model, withCxt = loadModel(opt.modelIdx)

local modelPrefix = opt.dataOut -- paths.concat(opt.dataOut, opt.modelIdx)
if not paths.dirp(modelPrefix) and not paths.mkdir(modelPrefix) then
   cmd:error('Error / unable to find models in: ' .. modelPrefix .. '\n')
end

for n, sample in data:run() do

  local dir = paths.concat(modelPrefix, paths.dirname(sample.path))
  if not paths.dirp(dir) and not paths.mkdir(dir) then
    cmd:error('Error / unable to create output directory: ' .. dir .. '\n')
  end

  local height, width = sample.orig:size(2), sample.orig:size(3)

  extMem:zero()

  if opt.gpu then sample.input = sample.input:cuda() end

  io.write(sample.path .. ' -> ')
  if withCxt then cxtMem:zero() end

  local cxtIn = {0, sample.input, cxtMem}
  local ctlIn = {sample.input, extMem, 0}

  local actIn = {}
  for j = 1,opt.nStates do
    states[j]:zero()
    table.insert(actIn, states[j])
  end

  local decIn = {0, sample.input}

  table.insert(actIn, sample.input)
  table.insert(actIn, extMem)
  if withCxt then table.insert(actIn, cxtMem) end

  -- ppc
  local ppcOut = model.ppc:forward(sample.input)
  local aMask = spSoftMax:forward(ppcOut[1])
  local fMask = ppcOut[2]

  table.insert(actIn, aMask)
  table.insert(actIn, fMask)

  table.insert(decIn, aMask)
  table.insert(decIn, fMask)

  table.insert(ctlIn, aMask)
  table.insert(ctlIn, fMask)

  table.insert(cxtIn, aMask)
  table.insert(cxtIn, fMask)

  ---
  -- Magick Loop
  ---
  local msks = {}
  local actPrev

  local seqDir = getDirFromPath(dir, sample.path, '_seq')
  local attMasks = {}

  for t = 1,opt.maxLength do

    -- act & dec
    local actOut = model.act:forward(actIn)
    decIn[1] = actOut[1]
    local msk = model.dec:forward(decIn)

    --local fileOut = string.gsub(sample.path, '_rgb', string.format('_label_%02d', t))
    --local finalPath = paths.concat(modelPrefix, fileOut)
    --image.save(finalPath, msk[1])
    --io.read()

    -- ctl
    local lastState = actOut[1 + opt.nStates]
    local ctlOut = model.ctl:forward(lastState)
    local ctlVal, ctlIdx = ctlOut[1]:max(1)
    local ctlVals = softMax:forward(ctlOut)
    --if (ctlIdx[1] == 1 or ctlVals[1][1] > 0.3) and t > 1  then
    if ctlIdx[1] == 1 and t > 1  then
      --print('Break', ctlVals[1][1])
      break
    else
      --print(ctlVals[1][1])
      table.insert(msks, msk[1]:float())
    end

    -- flash
    if opt.vi > 0 then
      if actPrev then
        local attMask = interpolate(actOut[1], actPrev, decIn, model.dec, opt.vi, fMask)
        table.insert(attMasks, attMask)
      end
      actPrev = actOut[1]:clone()
    end

    for j = 1,opt.nStates do actIn[j]:copy(actOut[j + 1]) end

    -- cxt
    if withCxt and opt.enCxt then
      cxtIn[1] = msk
      cxtMem:copy(model.cxt:forward(cxtIn))
    end

    extMem:cmax(msk)
  end

  ---
  -- Saving
  ---
  local mskTensor = torch.Tensor(#msks, height, width)
  local canvasFg = torch.zeros(height, width)

  for m = 1,#msks do
    -- undo DA
    local maskFull = image.scale(msks[m][1], width, height, 'bilinear')

    if opt.pps then
      -- bilateral filter
      maskFull = cv.bilateralFilter{src = maskFull, d = 5, sigmaColor = 10, sigmaSpace = 10}

      -- dilation
      --maskFull = cv.dilate{src = maskFull, kernel = dKernel}
    end

    if opt.vi > 0 and m < #msks then
      local att = attMasks[m]
      for i = 1,#att do
        att[i] = image.scale(att[i]:squeeze(), width, height, 'bilinear')
      end
    end

    if opt.vraw then
      local fnSuffix = string.format('_mask_%02d', m)
      local fnMask = string.gsub(sample.path, '_rgb', fnSuffix)
      local out_filename = paths.concat(modelPrefix, fnMask)
      image.save(out_filename, maskFull)
    end

    local canvasMsk = torch.gt(maskFull, opt.cutoff):float()

    mskTensor[m]:copy(maskFull:cmul(canvasMsk))
    canvasFg:cmax(canvasMsk)
  end

  if opt.exportNpy then
    local fOut = string.gsub(sample.path, '_rgb.png', '_numpy.npy')
    local fPath = paths.concat(modelPrefix, fOut)
    npy4th.savenpy(fPath, mskTensor)
  end

  local finalFg, finalMask = mskTensor:max(1)
  local fileOut = string.gsub(sample.path, '_rgb', '_labels')
  local finalPath = paths.concat(modelPrefix, fileOut)

  finalMask = finalMask:float()
  finalMask:cmul(canvasFg)

  for m = 1,#msks do
    mskTensor[m]:copy(torch.eq(finalMask, m))
  end

  if opt.vo then
    local rgbMask = drawMasks(sample.orig, mskTensor, nil, opt.alpha)
    local fileRGB = string.gsub(sample.path, '_rgb', '_order')
    local out_filename = paths.concat(modelPrefix, fileRGB)
    image.save(out_filename, rgbMask)

    if opt.vi > 0 and #attMasks > 0 then
      makeVideoSequence(seqDir, sample.orig, attMasks, rgbMask, opt.alpha, opt.cutoff)
    end
  end

  if opt.vr then
    local rgbMask = drawMasks(sample.orig, mskTensor, nil, opt.alpha, nil, true)
    local fileRGB = string.gsub(sample.path, '_rgb', '_random')
    local out_filename = paths.concat(modelPrefix, fileRGB)
    image.save(out_filename, rgbMask)
  end

  if opt.vc then
    local rgbMask = drawMasks(sample.orig, mskTensor, labels, opt.alpha)
    local fileRGB = string.gsub(sample.path, '_rgb', '_semantic')
    local out_filename = paths.concat(modelPrefix, fileRGB)
    image.save(out_filename, rgbMask)
  end

  io.write(finalPath .. ' #' .. (#msks) .. '\n')

  image.save(finalPath, finalMask:div(255))

  collectgarbage()
  collectgarbage()
end
