--
-- Evaluation Script for CVPPP leaf dataset
--
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'image'

--local cv = require 'cv'
--require 'cv.imgproc'


-- custom
local DataLoader = require 'cvppp_data'
require 'KLDCriterion'
require 'SamplerVAE'
require 'SpatialDownSampling'

package.path =  package.path .. ';../../?.lua;'

require 'TimeBatchNorm'
require 'utils/ContextManager'

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
cmd:option('-maxLength',      21, 'Maximum number of predictions')
cmd:option('-modelIdx',    '999', 'Model ID')
cmd:option('-v',           false, 'Visualisation')
cmd:option('-vi',              0, 'Visualise interpolation between the predictions with X steps')
cmd:option('-vv',          false, 'Visualise each mask individually')
cmd:option('-vgt',         false, 'Visualise ground truth')
cmd:option('-pps',         false, 'Post-processing')
cmd:option('-enCxt',       false, 'Enable CXT')
cmd:option('-cxtSize',         2, '1 for binary context; 2 for angle context;')
cmd:option('-withFg',      false, 'Use foreground')
cmd:option('-cutoff',        0.3, 'Cut-off value for mask binarisation')
cmd:option('-alpha',         0.6, '(Visualisation) Mask visibility')
cmd:option('-nThreads',        4, 'Number of data loading threads')

cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- constants
opt.Size = 224
opt.Height, opt.xHeight = 530, 237
opt.Width, opt.xWidth = 500, 224
opt.nStates = 4

-- initialising context update
opt.imWidth, opt.imHeight = 224, 224
cxtManager = ContextManager(opt, false)
local cxtUpdate = nil
if opt.cxtSize == 1 then
  cxtUpdate = function(x,y,z) cxtManager:max(x,y,z) end
elseif opt.cxtSize == 2 then
  cxtUpdate = function(x,y,z) cxtManager:angles(x,y,z) end
end

if opt.v then
  require 'utils'
end

--
-- Ordering info
--
local orderCor = torch.zeros(opt.maxLength, opt.maxLength) -- order correlation
local orderMap = torch.zeros(opt.Height, opt.Width)        -- order map
local orderMapCount = torch.zeros(opt.Height, opt.Width)   -- counts for averaging orderMap


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

  for i, net in pairs(model) do
    print('Net: ' .. i)
    for k, mod in pairs(net:findModules('cudnn.SpatialConvolution')) do
      mod:apply(function(m)
         if m.setMode then
           print('cudnn: Setting deterministic mode')
           m:setMode(1,1,1)
         end
      end)
    end
  end

  return model, withCxt
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
-- Reading the data
--
local data = DataLoader.create(opt)
local extMem = torch.Tensor(1, opt.cxtSize, opt.Size, opt.Size)
local prevMasks = torch.Tensor(1, 1, opt.Size, opt.Size)
local cxtMem = torch.Tensor(1, 3, opt.Size, opt.Size)
local state = torch.Tensor(1, 512)

--
-- Post-processing structs
--
local dKernel = torch.ones(5, 5)


if opt.gpu then
  extMem = extMem:cuda()
  prevMasks = prevMasks:cuda()
  cxtMem = cxtMem:cuda()
  state = state:cuda()
  cxtManager:copyCuda()
end

local states = {}
for i = 1,opt.nStates do table.insert(states, state:clone()) end

local spSoftMax = cudnn.SpatialSoftMax()
if opt.gpu then spSoftMax = spSoftMax:cuda() end


local softmax = nn.SoftMax()
if opt.gpu then softmax = softmax:cuda() end

---
-- Main Program
---
local model, withCxt = loadModel(opt.modelIdx)

local modelPrefix = opt.dataOut --paths.concat(opt.dataOut, opt.modelIdx)
if not paths.dirp(modelPrefix) and not paths.mkdir(modelPrefix) then
   cmd:error('Error / unable to find models in: ' .. modelPrefix .. '\n')
end

for n, sample in data:run() do

  local dir = paths.concat(modelPrefix, paths.dirname(sample.path))
  if not paths.dirp(dir) and not paths.mkdir(dir) then
    cmd:error('Error / unable to create output directory: ' .. dir .. '\n')
  end

  extMem:zero()
  prevMasks:zero()

  if opt.gpu then sample.input = sample.input:cuda() end

  io.write(sample.path .. ' -> ')
  if withCxt then cxtMem:zero() end

  local cxtIn = {0, sample.input, cxtMem}

  local actIn = {}
  for j = 1,opt.nStates do
    table.insert(actIn, states[j]:zero())
  end

  local decIn = {0, sample.input}

  table.insert(actIn, sample.input)
  table.insert(actIn, extMem)
  if withCxt then table.insert(actIn, cxtMem) end

  -- ppc
  local ppcOut = model.ppc:forward(sample.input)
  local aMask = spSoftMax:forward(ppcOut[1])
  local fMask = ppcOut[2]
  if not sample.fg then sample.fg = fMask end

  if opt.gpu then sample.fg = sample.fg:cuda() end

  table.insert(actIn, aMask)
  table.insert(actIn, fMask)

  table.insert(decIn, aMask)
  table.insert(decIn, fMask)

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
    local msk = model.dec:forward(decIn):clone()

    if opt.withFg and sample.fg then
      msk:cmul(sample.fg)
    end

    --local fileOut = string.gsub(sample.path, '_rgb', string.format('_label_%02d', t))
    --local finalPath = paths.concat(modelPrefix, fileOut)
    --image.save(finalPath, msk[1])
    --io.read()

    -- ctl
    local lastState = actOut[1 + opt.nStates]
    local ctlOut = model.ctl:forward(lastState)
    local ctlVal, ctlIdx = ctlOut[1]:max(1)
    --ctlOut = softmax:forward(ctlOut)
    if ctlOut[1][1] > 0.7 then
      break
    else
      table.insert(msks, msk:float())
    end

    -- flash
    if opt.vi > 0 then
      if actPrev then
        local attMask = interpolate(actOut[1], actPrev, decIn, model.dec, opt.vi, sample.fg)
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

    cxtUpdate(extMem, msk, prevMasks)
  end

  ---
  -- Saving
  ---
  local mskTensor = torch.zeros(#msks, opt.Height, opt.Width)
  local canvas = torch.zeros(opt.xHeight, opt.xWidth)
  local canvasFg = torch.zeros(opt.Height, opt.Width)
  local mskSize = torch.Tensor(#msks)

  local hOff = math.ceil((opt.xHeight - opt.Size)/2)
  for m = 1,#msks do
    -- undo DA
    canvas:sub(hOff + 1, hOff + opt.Size):copy(msks[m])
    local canvasFull = image.scale(canvas, opt.Width, opt.Height, 'bilinear')

    if opt.vi > 0 and m < #msks then
      local att = attMasks[m]
      for i = 1,#att do
        local attCanvas = torch.zeros(opt.xHeight, opt.xWidth)
        attCanvas:sub(hOff + 1, hOff + opt.Size):copy(att[i])
        att[i] = image.scale(attCanvas, opt.Width, opt.Height, 'bilinear')
      end
    end

    --
    -- post-processing: bilateral filter
    --
    if opt.pps then
      -- bilateral filter
      canvasFull = cv.bilateralFilter{src = canvasFull, d = 5, sigmaColor = 10, sigmaSpace = 10}

      -- dilation
      --canvasFull = image.dilate(canvasFull, dKernel)
      --canvasFull = cv.dilate{src = canvasFull, kernel = dKernel}
    end

    local canvasMsk = torch.gt(canvasFull, opt.cutoff):float()

    canvasFg:cmax(canvasMsk)
    canvasFull:cmul(canvasMsk)

    mskTensor[m]:copy(canvasFull)
    mskSize[m] = canvasMsk:sum()
  end

  --
  -- Ordering the mask size for correlation
  --
  local mSize, mIdx = torch.sort(mskSize, 1, true)
  for m = 1,#msks do
    orderCor[m][mIdx[m]] = orderCor[m][mIdx[m]] + 1
  end

  --
  -- enforcing one label
  --
  local predFg, finalMask = mskTensor:max(1)
  finalMask = finalMask:float():cmul(canvasFg)

  orderMap:add(finalMask)
  orderMapCount:add(finalMask:gt(0):float())

  local fileOut = string.gsub(sample.path, '_rgb', '_label')
  local finalPath = paths.concat(modelPrefix, fileOut)

  if opt.v then
    local filergb = string.gsub(sample.path, '_rgb', '_label_c')
    local out_filename = paths.concat(modelPrefix, filergb)

    local maskTensor = getMaskTensor(finalMask)
    local rgbMask = drawMasks(sample.orig, maskTensor, nil, opt.alpha)
    image.save(out_filename, rgbMask)

    if opt.vi > 0 then
      makeVideoSequence(seqDir, sample.orig, attMasks, rgbMask, opt.alpha, opt.cutoff)
    end

    if opt.vgt then
      local gtMaskTensor = getMaskTensor(sample.target)
      local gtMask = drawMasks(sample.orig, gtMaskTensor, nil, opt.alpha, nil, true)
      local gtFile = string.gsub(sample.path, '_rgb', '_gt')
      local out_filename = paths.concat(modelPrefix, gtFile)
      image.save(out_filename, gtMask)
    end

    if opt.vv then
      for m = 1,#msks do
        local fileMsk = string.gsub(sample.path, '_rgb', string.format('_label_%02d', m))
        fileMsk = paths.concat(modelPrefix, fileMsk)
        image.save(fileMsk, msks[m][1])
      end
    end
  end

  io.write(finalPath .. ' #' .. (#msks) ..  '\n')
  finalMask:div(255)
  image.save(finalPath, finalMask)
end

if opt.v then

  function mapRGB(mat, scheme, blackout)
    local h, w = unpack(mat:size():totable())
    local nz = mat:gt(0)

    colormap:setSteps(mat:max())
    colormap:setStyle(scheme)

    local orderCorRGB = colormap:convert(mat)
    nz = nz:typeAs(orderCorRGB):view(1, h, w):expandAs(orderCorRGB)

    if blackout then
      orderCorRGB:cmul(nz)
    end

    return orderCorRGB
  end

  -- Order correlation
  print("Plotting order correlation")
  local orderCorRGB = mapRGB(orderCor, 'jet', false)
  local fileMsk = paths.concat(modelPrefix, "order_correlation.png")
  orderCorRGB = image.scale(orderCorRGB, 512, 512, 'simple')
  image.save(fileMsk, orderCorRGB)

  -- Order map
  print("Plotting order map")
  orderMapCount[orderMapCount:eq(0)] = 1e-8
  orderMap:cdiv(orderMapCount):round()
  local orderMapRGB = mapRGB(orderMap, 'jet', true)
  local fileMsk = paths.concat(modelPrefix, "order_map.png")
  image.save(fileMsk, orderMapRGB)
end
