local nn = require 'nn'
require 'cunn'
require 'nngraph'

local function createModel(opt)

  function addLayer(model, nChannels, opt)
    if opt.optMemory >= 3 then
       model:add(nn.DenseConnectLayerCustom(nChannels, opt))
    else
       model:add(DenseConnectLayerStandard(nChannels, opt))     
    end
  end

  function addTransitionUp(model, nChannels, nChannelsOut)
    if opt.optMemory >= 3 then     
       model:add(nn.JoinTable(2))
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    --if opt.decoder == 'deconv' then
    model:add(cudnn.SpatialFullConvolution(nChannels, nChannelsOut, 3,3, 2,2, 1,1, 1,1))
    --elseif opt.decoder == 'bilinear' then
    --  model:add(nn.SpatialUpSamplingBilinear(2))
    --  model:add(cudnn.SpatialConvolution(nChannels, nChannelsOut, 3,3, 1,1, 1,1, 1,1))
    --end

    if opt.dropRate > 0 then model:add(nn.Dropout(opt.dropRate)) end
  end

  local function addDenseBlock(model, nChannels, opt, N)
    for i = 1, N do 
       addLayer(model, nChannels, opt)
       nChannels = nChannels + opt.growthRate
    end
    return nChannels
  end

  function decode(opt, inputDec, context, nCxtChannels, nChannelsEnc)
    local mask
    local action = inputDec[1]
    if #inputDec == 1 then action = nn.SelectTable(1)(action) end

    print("Initialising _" .. opt.decVer .. "_ decoder")
    if opt.decVer == 'simple' or opt.decVer == 'resnet' then
      if opt.dataset == 'mnist' or opt.dataset == 'mathnist' or opt.actorVer == 'resnet_dense' then
        stages = {64, 48, 32, 16}
      else
        stages = {96, 64, 48, 32, 16}
      end

      local channelsInS = nChannelsEnc + nCxtChannels
      local out = action
      for s = 1,#stages do
        if context[s] then out = nn.JoinTable(1, 3)({out, context[s]}) end

        out = nn.SpatialUpSamplingNearest(2)(out)
        out = nn.SpatialReplicationPadding(1, 1, 1, 1)(out)
        out = cudnn.SpatialConvolution(channelsInS, stages[s], 3,3)(out)
        out = nn.LeakyReLU(true)(out)
        if opt.decVerMod == 'extra' then
          out = nn.SpatialReplicationPadding(1, 1, 1, 1)(out)
          out = cudnn.SpatialConvolution(stages[s], stages[s], 3,3)(out)
          out = nn.LeakyReLU(true)(out)
        end
        channelsInS = stages[s] + nCxtChannels
      end

      if #inputDec > 1 then
        local decLast = {}
        for i = 2,#inputDec do
          table.insert(decLast, inputDec[i])
        end

        -- last conv with original-sized images
        out = nn.JoinTable(1, 3)({out, unpack(decLast)})
      end

      if opt.semSeg then -- semantic segmentation mode
        local numOut = opt.numClasses
        mask = nn.SpatialConvolution(channelsInS, numOut, 1,1)(out)
      else
        out = nn.SpatialConvolution(channelsInS, 1, 1,1)(out)
        mask = nn.Sigmoid(true)(out)
      end
    elseif opt.decVer == 'softmax' then
      stages = {96, 64, 48, 32, 16}
      local spStages = {2, 4, 8, 16, 32}

      local channelsInS = nChannelsEnc + nCxtChannels
      local sp = {action} -- contains spatial softmax tensors
      local out = action
      local featOut
      local chSoftMax = opt.latentSize
      for s = 1,#stages do

        -- thresholding
        out = nn.CAdd(1, chSoftMax, 1, 1)(out)
        out = nn.LeakyReLU(true)(out)

        local channelsIn = channelsInS
        if featOut then
          out = nn.JoinTable(1, 3)({featOut, out})
          channelsIn = channelsIn + stages[s - 1]
        end

        -- concatentating location with context
        if context[s] then out = nn.JoinTable(1, 3)({out, context[s]}) end

        if opt.decVerUp == 'nearest' then
          print("Upsampling: nearest neighbour")
          out = nn.SpatialUpSamplingNearest(2)(out)
          out = nn.SpatialReplicationPadding(1, 1, 1, 1)(out)
          out = cudnn.SpatialConvolution(channelsIn, stages[s], 3,3)(out)
          out = nn.SpatialDropout(0.5)(out)
        elseif opt.decVerUp == 'bilinear' then
          print("Upsampling: bilinear")
          out = nn.SpatialUpSamplingBilinear(2)(out)
          out = nn.SpatialReplicationPadding(1, 1, 1, 1)(out)
          out = cudnn.SpatialConvolution(channelsIn, stages[s], 3,3)(out)
          out = nn.SpatialDropout(0.5)(out)
        else
          print("Upsampling: deconv 2x2")
          out = cudnn.SpatialFullConvolution(channelsIn, stages[s], 2,2, 2,2)(out)
        end

        --out = nn.SpatialBatchNormalization(stages[s])(out)

        if opt.decVerMod == 'extra' then
          out = nn.LeakyReLU(true)(out)
          out = nn.SpatialReplicationPadding(1, 1, 1, 1)(out)
          out = cudnn.SpatialConvolution(stages[s], stages[s], 3,3)(out)
          out = nn.SpatialDropout(0.5)(out)
          --out = nn.SpatialBatchNormalization(stages[s])(out)
        end

        featOut = nn.LeakyReLU(true)(out)
        out = cudnn.SpatialConvolution(stages[s], spStages[s], 1,1)(featOut)

        for si = 1,#sp do
          sp[si] = nn.SpatialUpSamplingBilinear(2)(sp[si])
        end

        table.insert(sp, out)

        -- concatentating previous locations
        out = nn.JoinTable(1, 3)(sp)
        channelsInS = channelsInS + spStages[s]
        chSoftMax = chSoftMax + spStages[s]

        -- applying spatial softmax
        out = nn.LocationLogSoftMax(true)(out)
      end

      out = nn.CAdd(1, chSoftMax, 1, 1)(out)
      out = nn.LeakyReLU(true)(out)

      if #inputDec > 1 then
        local decLast = {}
        for i = 2,#inputDec do
          table.insert(decLast, inputDec[i])
        end

        -- last conv with original-sized images
        out = nn.JoinTable(1, 3)({out, featOut, unpack(decLast)})
        channelsInS = channelsInS + stages[#stages]
      end

      out = nn.SpatialConvolution(channelsInS, 1, 1,1)(out)
      mask = nn.Sigmoid(true)(out)
    elseif opt.decVer == 'densenet' then
      stages = {8, 12, 6, 4, 2}

      local channelsInS = nChannelsEnc + nChannelsIn
      local out = action
      for s = 1,#stages do
        out = nn.JoinTable(1, 3)({out, context[s]})
        local seq = nn.Sequential()
        nChannels = addDenseBlock(seq, channelsInS, opt, stages[s])
        nChannelsOut = math.floor(nChannels*opt.reduction)
        print('Transition Up ' .. nChannels .. ' -> ' .. nChannelsOut)
        addTransitionUp(seq, nChannels, nChannelsOut)
        out = seq(out)
        channelsInS = nChannelsOut + nChannelsIn
      end

      local decLast = {}
      for i = 2,#inputDec do
        table.insert(decLast, inputDec[i])
      end

      out = nn.JoinTable(1, 3)({out, unpack(decLast)})
      local seq = nn.Sequential()
      nChannels = addDenseBlock(seq, channelsInS, opt, 1)
      seq:add(nn.JoinTable(2))
      seq:add(cudnn.SpatialBatchNormalization(nChannels))
      seq:add(cudnn.ReLU(true))
      seq:add(cudnn.SpatialConvolution(nChannels, 1, 1,1))

      mask = nn.Sigmoid()(seq(out))
    else
      print('Unknown actor version: ' .. opt.decVer)
      os.exit(1)
    end
    return mask
  end

  function buildPyramid(opt, x, nCh, n)
    local n = n or 5
    local outs = {}
    for i = 1,n do
      table.insert(outs, 1, nn.SpatialDownSampling(nCh, 4,4, 2,2, 1,1)(x))
      x = outs[1]
    end
    return outs
  end

  local nCxtChannels = 0
  local action = nn.Identity()()

  local inputsNet = {action}

  local dStages = 5
  if opt.dataset == 'mnist' or opt.dataset == 'mathnist' or opt.actorVer == 'resnet_dense' then
    dStages = 4
  end

  local contextPyramid = {}
  for i = 1,dStages do contextPyramid[i] = {} end

  local imagePyramid
  if opt.pyramidImage then
    local image = nn.Identity()()
    table.insert(inputsNet, image)
    nCxtChannels = nCxtChannels + opt.imageCh
    imagePyramid = buildPyramid(opt, image, opt.imageCh, dStages)
    for i = 1,#contextPyramid do table.insert(contextPyramid[i], imagePyramid[i]) end
  end

  local angleMask
  if opt.pyramidAngles then
    angleMask = nn.Identity()()
    table.insert(inputsNet, angleMask)
    nCxtChannels = nCxtChannels + opt.numAngles
    local anglePyramid = buildPyramid(opt, angleMask, opt.numAngles, dStages)
    for i = 1,#contextPyramid do table.insert(contextPyramid[i], anglePyramid[i]) end
  end
  
  local fgMask
  if opt.pyramidFg then
    fgMask = nn.Identity()()
    table.insert(inputsNet, fgMask)
    nCxtChannels = nCxtChannels + 1
    local fgPyramid = buildPyramid(opt, fgMask, 1, dStages)
    for i = 1,#contextPyramid do table.insert(contextPyramid[i], fgPyramid[i]) end
  end

  -- Joining the context pyramid
  for i = 1,#contextPyramid do
    if #contextPyramid[i] > 1 then
      contextPyramid[i] = nn.JoinTable(1, 3)(contextPyramid[i])
    elseif #contextPyramid[i] == 1 then
      contextPyramid[i] = contextPyramid[i][1]
    else
      contextPyramid[i] = nil
    end
  end

  local maskOut = decode(opt, inputsNet, contextPyramid, nCxtChannels, opt.latentSize)
  local model = nn.gModule(inputsNet, {maskOut})

  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
      else
        v.bias:zero()
      end
    end
  end

  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
       v.weight:fill(1)
       v.bias:zero()
    end
  end

  BNInit('cudnn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')

  -- the weights for nn.Linear are initialised in
  -- the constructor
  for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end

  if opt.nGPU > 0 then
    model:cuda()
    if opt.cudnn == 'deterministic' then
       model:apply(function(m)
          if m.setMode then m:setMode(1,1,1) end
       end)
    end
  end

  return model
end


return createModel
