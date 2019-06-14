local nn = require 'nn'
require 'cunn'
require 'nngraph'
require 'modules/TimeBatchNorm'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

-- parameters are not learned
local SBatchNorm = TimeBatchNorm

local function createModel(opt)

  function addLayer(model, nChannels, opt)
    if opt.optMemory >= 3 then
       model:add(nn.DenseConnectLayerCustom(nChannels, opt))
    else
       model:add(DenseConnectLayerStandard(nChannels, opt))     
    end
  end

  function addTransition(model, nChannels, nOutChannels, opt)
    if opt.optMemory >= 3 then     
       model:add(nn.JoinTable(2))
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))      
    model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
    if opt.dropRate > 0 then model:add(nn.Dropout(opt.dropRate)) end
    model:add(cudnn.SpatialAveragePooling(2, 2))
  end

  local function addDenseBlock(model, nChannels, opt, N)
    for i = 1, N do 
       addLayer(model, nChannels, opt)
       nChannels = nChannels + opt.growthRate
    end
    return nChannels
  end

  local function LSTM(opt, x, state)
    local outputs = {}
    for L = 1,opt.rnn_layers do
      -- c,h from previos timesteps
      local prev_c = state[L*2-1]
      local prev_h = state[L*2]

      -- the input to this layer
      local i2h = nn.Linear(opt.rnnSize, 4*opt.rnnSize)(x)
      local h2h = nn.Linear(opt.rnnSize, 4*opt.rnnSize)(prev_h)

      i2h.data.module.bias:fill(1e-4)
      i2h.data.module.weight:uniform(-1e-3, 1e-3)

      h2h.data.module.bias:fill(1e-4)
      h2h.data.module.weight:uniform(-1e-3, 1e-3)
      
      --for n = 1,4 do
      --  i2h.data.module.weight:sub((n - 1)*opt.rnnSize + 1, n*opt.rnnSize):eye(opt.rnnSize)
      ----  h2h.data.module.weight:sub((n - 1)*opt.rnnSize + 1, n*opt.rnnSize):eye(opt.rnnSize)
      --end

      local all_input_sums = nn.CAddTable()({i2h, h2h})

      local reshaped = nn.Reshape(4, -1)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(1, 2)(reshaped):split(4)

      -- gates
      local ig = nn.Sigmoid()(n1)
      local fg = nn.Sigmoid()(n2)
      local og = nn.Sigmoid()(n3)

      -- decode the write inputs
      local z = nn.Tanh()(n4)

      -- perform the LSTM update
      local next_c = nn.CAddTable()({nn.CMulTable()({fg, prev_c}),
                                     nn.CMulTable()({ig, z})})

      -- gated cells form the output
      local next_h = nn.CMulTable()({og, nn.Tanh()(next_c)})

      table.insert(outputs, next_c)
      table.insert(outputs, next_h)

      x = next_h
      --if opt.dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end
    
    return outputs
  end

  function encode(opt, inputs, inChannels, input_cxt)
    local out_first = nn.JoinTable(1, 3)(inputs)

    local out
    if opt.actorVer ~= 'vgg16_imagenet' then
      out = cudnn.SpatialConvolution(inChannels, 32, 3,3, 1,1, 1,1)(out_first)
      if opt.withContext then
        print('Adding Context')
        local cxt = nn.Sigmoid(true)(input_cxt)
        cxt = cudnn.SpatialConvolution(opt.contextDim, 32, 3,3, 1,1, 1,1):noBias()(cxt)
        out = nn.CAddTable()({out, cxt})
      end

      --model:add(cudnn.SpatialBatchNormalization(32))
      out = cudnn.ReLU(true)(out)
    end

    local nChannels
    if opt.actorVer == 'densenet_pre' then
      print('   > Using pre-trained model')
      local dnet = torch.load('models/DenseNet/densenet-121.t7')

      -- surgery to adapt the 1st layer to a +1 channel
      local layer1 = dnet:get(1)
      local layerX = nn.SpatialConvolution(inChannels, 64, 7,7, 2,2, 3,3)
      layerX.weight:sub(1,-1, 1,3):copy(layer1.weight)
      layerX.weight:sub(1,-1, 4,3+opt.numAngles):uniform(-1e-4, 1e-4)

      dnet:remove(1)
      dnet:insert(layerX, 1)

      -- removing the last linear layer
      dnet:remove(77)
      dnet:remove(77)
      dnet:remove(77)

      out = dnet(out)
      
      -- out = nn.PrintSize()(out)
      nChannels = 1024
    elseif opt.actorVer == 'simple' then
      print('> Creating a simple net from scratch')

      local model = nn.Sequential()

      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      model:add(cudnn.SpatialConvolution(32, 48, 3,3, 1,1, 1,1))
      --model:add(cudnn.SpatialBatchNormalization(48))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      model:add(cudnn.SpatialConvolution(48, 64, 3,3, 1,1, 1,1))
      --model:add(cudnn.SpatialBatchNormalization(64))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      model:add(cudnn.SpatialConvolution(64, 96, 3,3, 1,1, 1,1))
      --model:add(cudnn.SpatialBatchNormalization(96))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      nChannels = 96 --opt.featureSize

      if opt.dataset ~= 'mnist' and opt.dataset ~= 'mathnist' then
        model:add(cudnn.SpatialConvolution(96, 128, 3,3, 1,1, 1,1))
        --model:add(cudnn.SpatialBatchNormalization(opt.featureSize))
        model:add(cudnn.ReLU(true))
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nChannels = 128
      end

      out = model(out)

    elseif opt.actorVer == 'vgg16_imagenet' then
      print('   > Using VGG16 pre-trained model')
      local vgg = torch.load('models/vgg/imgnet_VGG16.t7')

      -- surgery to adapt the 1st layer to a +1 channel
      local layer1 = vgg:get(1)
      local layerX = cudnn.SpatialConvolution(inChannels, 64, 3,3, 1,1, 1,1)
      layerX.weight:normal(0, 1)
      layerX.weight:sub(1,-1, 1, 3):copy(layer1.weight)

      vgg:remove(1)
      vgg:insert(layerX, 1)

      nChannels = 512

      -- removing the last 8 layers
      for n = 1,8 do
        vgg:remove()
      end

      out = vgg(out_first)
    elseif opt.actorVer == 'vgg16' then
      print('> Creating a VGG-16/2 net from scratch')
      local cfg = {'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}

      local model = nn.Sequential()
      model:add(cudnn.SpatialConvolution(32, 32, 3,3, 1,1, 1,1))

      nChannels = 32
      for k,v in ipairs(cfg) do
         if v == 'M' then
            model:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v / 2;
            local conv3 = cudnn.SpatialConvolution(nChannels,oChannels,3,3,1,1,1,1);
            model:add(conv3)
            model:add(cudnn.ReLU(true))
            nChannels = oChannels;
         end
      end

      out = model(out)
    elseif opt.actorVer == 'resnet_dense' then
			local depth = 34
			local shortcutType = 'B'
		  local iChannels

      print('Using ResNet-' .. depth)

		  -- The shortcut layer is either identity or 1x1 convolution
		  local function shortcut(nInputPlane, nOutputPlane, stride)
		   	local useConv = shortcutType == 'C' or
		   		 (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
		   	if useConv then
		   		 -- 1x1 convolution
		   		 return nn.Sequential()
		   				:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
		   				:add(SBatchNorm(opt, nOutputPlane))
		   	elseif nInputPlane ~= nOutputPlane then
		   		 -- Strided, zero-padded identity shortcut
		   		 return nn.Sequential()
		   				:add(nn.SpatialAveragePooling(1, 1, stride, stride))
		   				:add(nn.Concat(2)
		   					 :add(nn.Identity())
		   					 :add(nn.MulConstant(0)))
		   	else
		   		 return nn.Identity()
		   	end
		  end

		  -- The basic residual layer block for 18 and 34 layer network, and the
		  -- CIFAR networks
		  local function basicblock(n, stride)
		   	local nInputPlane = iChannels
		   	iChannels = n

		   	local s = nn.Sequential()
		   	s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
		   	s:add(SBatchNorm(opt, n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,1,1,1,1))
		   	s:add(SBatchNorm(opt, n))

		   	return nn.Sequential()
		   		 :add(nn.ConcatTable()
		   				:add(s)
		   				:add(shortcut(nInputPlane, n, stride)))
		   		 :add(nn.CAddTable(true))
		   		 :add(ReLU(true))
		  end

		  -- The bottleneck residual layer for 50, 101, and 152 layer networks
		  local function bottleneck(n, stride)
		   	local nInputPlane = iChannels
		   	iChannels = n * 4

		   	local s = nn.Sequential()
		   	s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
		   	s:add(SBatchNorm(opt, n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,stride,stride,1,1))
		   	s:add(SBatchNorm(opt, n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n*4,1,1,1,1,0,0))
		   	s:add(SBatchNorm(opt, n * 4))

		   	return nn.Sequential()
		   		 :add(nn.ConcatTable()
		   				:add(s)
		   				:add(shortcut(nInputPlane, n * 4, stride)))
		   		 :add(nn.CAddTable(true))
		   		 :add(ReLU(true))
		  end

		  -- Creates count residual blocks with specified number of features
		  local function layer(block, features, count, stride)
		   	local s = nn.Sequential()
		   	for i=1,count do
		   		 s:add(block(features, i == 1 and stride or 1))
		   	end
		   	return s
		  end

      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      --assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      --local def, nFeatures, block = table.unpack(cfg[depth])
      local def, nFeatures, block = table.unpack(cfg[depth])

      iChannels = 64

      -- The ResNet ImageNet model
			local model = nn.Sequential()
      model:add(Convolution(32,64,7,7,2,2,3,3))
      model:add(SBatchNorm(opt, 64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))

      local garmon = nn.Sequential()
      garmon:add(layer(block, 512, def[4], 2))
      garmon:add(nn.SpatialUpSamplingNearest(2))

      model:add(nn.ConcatTable()
                      :add(garmon)
                      :add(nn.Identity()))
      model:add(nn.JoinTable(1, 3))
      model:add(Convolution(512 + 256, 512, 3,3, 1,1, 1,1))

      --model:add(nn.PrintSize())
      --model:add(Avg(7, 7, 1, 1))
      --model:add(nn.View(nFeatures):setNumInputDims(3))
      --model:add(nn.Linear(nFeatures, 1000))
		
			out = model(out)
			nChannels = nFeatures
    elseif opt.actorVer == 'resnet' then
			local depth = 34
			local shortcutType = 'B'
		  local iChannels

      print('Using ResNet-' .. depth)

		  -- The shortcut layer is either identity or 1x1 convolution
		  local function shortcut(nInputPlane, nOutputPlane, stride)
		   	local useConv = shortcutType == 'C' or
		   		 (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
		   	if useConv then
		   		 -- 1x1 convolution
           local conv = nn.Sequential()
                          :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
           
           if opt.actorBN then conv:add(SBatchNorm(opt, nOutputPlane)) end

		   		 return conv
		   	elseif nInputPlane ~= nOutputPlane then
		   		 -- Strided, zero-padded identity shortcut
		   		 return nn.Sequential()
		   				:add(nn.SpatialAveragePooling(1, 1, stride, stride))
		   				:add(nn.Concat(2)
		   					 :add(nn.Identity())
		   					 :add(nn.MulConstant(0)))
		   	else
		   		 return nn.Identity()
		   	end
		  end

		  -- The basic residual layer block for 18 and 34 layer network, and the
		  -- CIFAR networks
		  local function basicblock(n, stride)
		   	local nInputPlane = iChannels
		   	iChannels = n

		   	local s = nn.Sequential()
		   	s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
        if opt.actorBN then s:add(SBatchNorm(opt, n)) end
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,1,1,1,1))
        if opt.actorBN then s:add(SBatchNorm(opt, n)) end

		   	return nn.Sequential()
		   		 :add(nn.ConcatTable()
		   				:add(s)
		   				:add(shortcut(nInputPlane, n, stride)))
		   		 :add(nn.CAddTable(true))
		   		 :add(ReLU(true))
		  end

		  -- The bottleneck residual layer for 50, 101, and 152 layer networks
		  local function bottleneck(n, stride)
		   	local nInputPlane = iChannels
		   	iChannels = n * 4

		   	local s = nn.Sequential()
		   	s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
		   	if opt.actorBN then s:add(SBatchNorm(opt, n)) end
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,stride,stride,1,1))
		   	if opt.actorBN then s:add(SBatchNorm(opt, n)) end
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n*4,1,1,1,1,0,0))
		   	if opt.actorBN then s:add(SBatchNorm(opt, n * 4)) end

		   	return nn.Sequential()
		   		 :add(nn.ConcatTable()
		   				:add(s)
		   				:add(shortcut(nInputPlane, n * 4, stride)))
		   		 :add(nn.CAddTable(true))
		   		 :add(ReLU(true))
		  end

		  -- Creates count residual blocks with specified number of features
		  local function layer(block, features, count, stride)
		   	local s = nn.Sequential()
		   	for i=1,count do
		   		 s:add(block(features, i == 1 and stride or 1))
		   	end
		   	return s
		  end

      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      --assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      --local def, nFeatures, block = table.unpack(cfg[depth])
      local def, nFeatures, block = table.unpack(cfg[depth])

      iChannels = 64

      -- The ResNet ImageNet model
			local model = nn.Sequential()
      model:add(Convolution(32,64,7,7,2,2,3,3))
      if opt.actorBN then model:add(SBatchNorm(opt, 64)) end
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      --model:add(nn.PrintSize())
      --model:add(Avg(7, 7, 1, 1))
      --model:add(nn.View(nFeatures):setNumInputDims(3))
      --model:add(nn.Linear(nFeatures, 1000))
		
			out = model(out)
			nChannels = nFeatures
    elseif opt.actorVer == 'densenet' then
      print('> Creating the DenseNet from scratch')
      -- we're using a 121-layer net
      stages = {3, 6, 12, 8}   -- 516

      nChannels = 2 * opt.growthRate

      local model = nn.Sequential()

      --Initial transforms follow ResNet(224x224)
      model:add(cudnn.SpatialConvolution(inChannels, nChannels, 7,7, 2,2, 3,3))
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

      -- Dense-Block
      for i = 1,3 do
        nChannels = addDenseBlock(model, nChannels, opt, stages[i])
        local nChannelsIn = nChannels
        addTransition(model, nChannels, math.floor(nChannels*opt.reduction), opt)
        nChannels = math.floor(nChannels*opt.reduction)
        print('Transition Down ' .. nChannelsIn .. ' -> ' .. nChannels)
      end

      --Dense-Block 4 and transition (7x7)
      nChannels = addDenseBlock(model, nChannels, opt, stages[4])

      -- Instead of transition
      model:add(nn.JoinTable(2))
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))

      out = model(out)
    else
      print('Unknown actor version: ' .. opt.actorVer)
      os.exit(1)
    end

    return out, nChannels
  end

  function bottleneckFC(opt, features, nChannelsIn, rnnState)
    print('Bottleneck: FC')

    out = features
    out = nn.Reshape(nChannelsIn*opt.actionX*opt.actionY)(out)
  
    local rnnStateNext
    if opt.mode == 'pretrain' then
      print("-- Actor: pretrain mode")
      out = nn.Linear(nChannelsIn*opt.actionX*opt.actionY, opt.featureSize):noBias()(out)
    else
      print("-- Actor: creating LSTM")
      out = nn.Linear(nChannelsIn*opt.actionX*opt.actionY, opt.rnnSize)(out)

      local current = nn.LeakyReLU(true)(out)
      rnnStateNext = LSTM(opt, current, rnnState)
      local hidden = rnnStateNext[#rnnStateNext]
      out = nn.JoinTable(2)({hidden, current})
      out = nn.Linear(2*opt.rnnSize, opt.featureSize):noBias()(out)
    end

    out = nn.LeakyReLU(true)(out)

    out = nn.Linear(opt.featureSize, 2*opt.latentSize)(out)
    out = nn.Reshape(2, opt.latentSize)(out)
    local mean, logvar = nn.SplitTable(1, 2)(out):split(2)

    local kldiv = nn.KLDCriterion(opt.lambdaKL)({mean, logvar})
    local sampler = nn.SamplerVAE()(kldiv)

    out = nn.Linear(opt.latentSize, opt.featureSize)(sampler)
    out = nn.LeakyReLU(true)(out)

    out = nn.Linear(opt.featureSize, opt.latentSize*opt.actionY*opt.actionX)(out)
    out = nn.LeakyReLU(true)(out)

    out = nn.Reshape(opt.latentSize, opt.actionY, opt.actionX)(out)

    return out, kldiv, sampler, rnnStateNext --, sout
  end

  local rnnState = {}
  if opt.mode ~= 'pretrain' then
    for i = 1,opt.rnn_layers do
      table.insert(rnnState, nn.Identity()())
      table.insert(rnnState, nn.Identity()())
    end
  end

  local nChannelsIn = opt.imageCh + opt.cxtSize
  local image = nn.Identity()()
  local maskIn = nn.Identity()()

  local inputsNet = {}
  for i = 1,#rnnState do
    table.insert(inputsNet, rnnState[i])
  end

  table.insert(inputsNet, image)
  table.insert(inputsNet, maskIn)

  local inputsEnc = {image, maskIn}

  local context
  if opt.withContext then
    context = nn.Identity()()
    table.insert(inputsNet, context)
  end

  if opt.predictAngles then
    local angleMask = nn.Identity()()
    table.insert(inputsNet, angleMask)
    table.insert(inputsEnc, angleMask)
    nChannelsIn = nChannelsIn + opt.numAngles
  end

  if opt.predictFg then
    local fgMask = nn.Identity()()
    table.insert(inputsNet, fgMask)
    table.insert(inputsEnc, fgMask)
    nChannelsIn = nChannelsIn + 1
  end

  local prepool, nChannelsEnc = encode(opt, inputsEnc, nChannelsIn, context)
  local action, kldiv, sampler, rnnStateNext = bottleneckFC(opt, prepool, nChannelsEnc, rnnState)

  if opt.noise > 0 then
    print("-- Actor: adding white noise")
    action = nn.WhiteNoise(0, opt.noise)(action)
  end

  if opt.actType == 'softmax' then
    print("-- Actor: adding spatial softmax")
    action = nn.LocationLogSoftMax(true)(action)
  elseif opt.actType == 'softmax_cmul' then
    print("-- Actor: adding spatial softmax with CMul")
    local loc = nn.LocationLogSoftMax(false)(action)
    loc = nn.Exp()(loc)
    --local avgLoc = nn.Replicate(opt.latentSize, 1, 2)
    --                  (nn.Mean(1, 3)(nn.Exp()(loc)))
    action = nn.CMulTable()({loc, action})
  elseif opt.actType == 'softmax_cmul_one' then
    print("-- Actor: adding SINGLE spatial softmax with CMul")
    local sp = cudnn.SpatialConvolution(opt.latentSize, 1, 1,1)(action)
    sp = cudnn.ReLU(true)(sp)

    local loc = nn.LocationLogSoftMax(false)(sp)
    loc = nn.View(-1, opt.actionY, opt.actionX)(loc)
    loc = nn.Replicate(opt.latentSize, 1, 2)(nn.Exp()(loc))
    action = nn.CMulTable()({loc, action})
  end

  local modelOuts = {action}
  if rnnStateNext then
    for n = 1,#rnnStateNext do
      table.insert(modelOuts, rnnStateNext[n])
    end
  end

  local model = nn.gModule(inputsNet, modelOuts)

  local timeBatchNorm = model:findModules('TimeBatchNorm')
  local function setTimestep(m, t)
    for k,v in pairs(timeBatchNorm) do
       v:setTimestep(t)
    end
  end

  model.setTimestep = setTimestep

  -- for logging
  model.kldiv = kldiv
  model.sampler = sampler

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

  if opt.actorBN then
    print('-- Actor: initialising BatchNorm')
    BNInit('cudnn.SpatialBatchNormalization')
  end


  if opt.actorVer == 'resnet' then
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')

    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
  end

  -- the weights for nn.Linear are initialised in
  -- the constructor
  for k,v in pairs(model:findModules('nn.Linear')) do
    if v.bias then
      v.bias:zero()
    end
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
