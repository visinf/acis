--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Sigmoid = cudnn.Sigmoid
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)

  local function LSTM(opt, x, state)
    local outputs = {}

    local prev_c = state[1]
    local prev_h = state[2]
  
    -- the input to this layer
    local i2h = nn.Linear(512, 4*opt.rnnSize)(x)
    local h2h = nn.Linear(opt.rnnSize, 4*opt.rnnSize)(prev_h)
  
    i2h.data.module.bias:fill(1e-4)
    i2h.data.module.weight:uniform(-1e-3, 1e-3)
  
    h2h.data.module.bias:fill(1e-4)
    h2h.data.module.weight:uniform(-1e-3, 1e-3)

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
    
    return outputs
  end
  
  -- RNN states
  local out

  local model
  local nChannels = 2 * opt.growthRate

  local image = nn.Identity()()
  local mask = nn.Identity()()
  local mem = nn.Identity()()
  local inputs = {image, mem, mask}

  local iDims = opt.imageCh + opt.cxtSize + 1

  if opt.criticGT then
      iDims = iDims + 2
      local maskAngles = nn.Identity()()
      table.insert(inputs, maskAngles)
  else
    if opt.predictAngles then
      iDims = iDims + opt.numAngles
      local angles = nn.Identity()()
      table.insert(inputs, angles)
    end

    if opt.predictFg then
      iDims = iDims + 1
      local fg = nn.Identity()()
      table.insert(inputs, fg)
    end
  end

  if opt.criticVer == 'lstm' then
    local rnnStates = {nn.Identity()(), nn.Identity()()}
    local inputs = {}
    for i = 1,#rnnStates do table.insert(inputs, rnnStates[i]) end

    local image = nn.Identity()()
    local mask = nn.Identity()()
    local extmem = nn.Identity()()
    local inputs_pre = {image, mask, extmem}

    local iDims = 3 + 1 + 1
    
    if opt.predictAngles then
      iDims = iDims + opt.numAngles
      local angles = nn.Identity()()
      table.insert(inputs_pre, angles)
    end

    if opt.predictFg then
      iDims = iDims + 1
      local fg = nn.Identity()()
      table.insert(inputs_pre, fg)
    end

    for i = 1,#inputs_pre do table.insert(inputs, inputs_pre[i]) end

    local oDims = opt.criticDim
    local shapes = {oDims, 2*oDims, 3*oDims, 4*oDims, 5*oDims}
  
    model = nn.Sequential()
    model:add(nn.JoinTable(2))

    for i = 1,#shapes do
      model:add(nn.SpatialConvolution(iDims, shapes[i], 3, 3, 1, 1, 1, 1))
      model:add(nn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      iDims = shapes[i]
    end

    oDims = shapes[#shapes]
    nChannels = oDims*opt.actionX*opt.actionY

    model:add(nn.Reshape(nChannels))

    out = model(inputs_pre)

    out = nn.Linear(nChannels, 512)(out)
    local current = nn.LeakyReLU(true)(out)

    local rnnStatesNext = LSTM(opt, current, rnnStates)
    local hidden = rnnStatesNext[#rnnStatesNext]

    out = nn.JoinTable(2)({hidden, current})

    out = nn.Linear(2*512, 512)(out)
    out = nn.LeakyReLU(true)(out)

    out = nn.Linear(512, 1)(out)
    --out = nn.SoftPlus()(out)

    model = nn.gModule(inputs, {out, unpack(rnnStatesNext)})
  elseif opt.criticVer == 'irnn' or opt.criticVer == 'irnn_pool' then
    print('Critic is IRNN')

    local image = nn.Identity()()
    local mem = nn.Identity()()
    local mask = nn.Identity()()
    local memax = nn.Identity()()
    local inputs = {image, mem, mask, memax}
    local ins = {image, mask, memax}

    local iDims = 3 + 1 + 1

    if opt.criticGT then
        iDims = iDims + 2
        local maskAngles = nn.Identity()()
        table.insert(inputs, maskAngles)
        table.insert(ins, maskAngles)
    else
      if opt.predictAngles then
        iDims = iDims + opt.numAngles
        local angles = nn.Identity()()
        table.insert(inputs, angles)
        table.insert(ins, angles)
      end

      if opt.predictFg then
        iDims = iDims + 1
        local fg = nn.Identity()()
        table.insert(inputs, fg)
        table.insert(ins, fg)
      end
    end

    local irnn = cudnn.SpatialConvolution(opt.criticMem, opt.criticMem, 1,1)
    irnn.bias:zero()
    irnn.weight[{{},{},1,1}]:eye(opt.criticMem)

    -- updating the memory
    local out = nn.JoinTable(2)(ins)
    out = cudnn.SpatialConvolution(iDims, 32, 3,3, 1,1, 1,1)(out)
    out = nn.LeakyReLU(true)(out)

    out = cudnn.SpatialConvolution(32, opt.criticMem, 3,3, 1,1, 1,1)(out)
    
    local mem_in = irnn(mem)
    local mem_next = nn.LeakyReLU(true)(nn.CAddTable()({out, mem_in}))

    local oDims = opt.criticDim
    local shapes = {oDims, 2*oDims, 3*oDims, 4*oDims, 5*oDims}

    model = nn.Sequential()

    local dim_i = opt.criticMem
    for i = 1,#shapes do
      model:add(nn.SpatialConvolution(dim_i, shapes[i], 3, 3, 1, 1, 1, 1))
      model:add(nn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      dim_i = shapes[i]
    end

    if opt.criticVer == 'irnn' then
      nChannels = dim_i*opt.actionX*opt.actionY
    else
      nChannels = dim_i
      model:add(nn.SpatialAveragePooling(opt.actionX, opt.actionY))
    end

    model:add(nn.Reshape(nChannels))

    out = model(mem_next)

    out = nn.Linear(nChannels, 512)(out)
    out = nn.LeakyReLU(true)(out)

    local score = nn.Linear(512, 1)(out)
    --score = nn.SoftPlus()(score)

    model = nn.gModule(inputs, {score, mem_next})
  elseif opt.criticVer == 'resnet' then
			local depth = 34
			local shortcutType = 'B'
		  local iChannels

      print('Critic: Using ResNet-' .. depth)

		  -- The shortcut layer is either identity or 1x1 convolution
		  local function shortcut(nInputPlane, nOutputPlane, stride)
		   	local useConv = shortcutType == 'C' or
		   		 (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
		   	if useConv then
		   		 -- 1x1 convolution
		   		 return nn.Sequential()
		   				:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
		   				:add(SBatchNorm(nOutputPlane))
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
		   	s:add(SBatchNorm(n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,1,1,1,1))
		   	s:add(SBatchNorm(n))

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
		   	s:add(SBatchNorm(n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n,3,3,stride,stride,1,1))
		   	s:add(SBatchNorm(n))
		   	s:add(ReLU(true))
		   	s:add(Convolution(n,n*4,1,1,1,1,0,0))
		   	s:add(SBatchNorm(n * 4))

		   	return nn.Sequential()
		   		 :add(nn.ConcatTable()
		   				:add(s)
		   				:add(shortcut(nInputPlane, n * 4, stride)))
		   		 :add(nn.CAddTable(true))
		   		 :add(ReLU(true))
		  end

      local function reduce(data, W, H, N)
        local out = nn.SpatialAveragePooling(W, H)(data)
        return nn.Reshape(N)(out)
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
			model = nn.Sequential()
      model:add(nn.JoinTable(2))
      model:add(Convolution(iDims,64,7,7,2,2,3,3))
      model:add(ReLU(true))
      --model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))

      out = model(inputs)

      local l2 = layer(block, 128, def[2], 2)
      local l3 = layer(block, 256, def[3], 2)
      local l4 = layer(block, 512, def[4], 2)

      if opt.criticType == 'softmax_cmul' then
        print("-- Critic: adding spatial softmax with CMul")
        local softmax = nn.Sequential()
                              :add(nn.LocationLogSoftMax(false))
                              :add(nn.Exp())
        model:add(nn.ConcatTable()
                        :add(softmax)
                        :add(nn.Identity()))
             :add(nn.CMulTable())
             :add(nn.Sum(3, 3))
             :add(nn.Sum(2, 2))
      elseif opt.criticType == 'avg' then
        print("-- Critic: adding global average pooling")
        s1 = reduce(out, 16*opt.actionX, 16*opt.actionY, 64)

        out2 = l2(out)
        s2 = reduce(out2, 8*opt.actionX, 8*opt.actionY, 128)

        out3 = l3(out2)
        s3 = reduce(out3, 4*opt.actionX, 4*opt.actionY, 256)

        out4 = l4(out3)
        s4 = reduce(out4, 2*opt.actionX, 2*opt.actionY, 512)

        nFeatures = nFeatures + 64 + 128 + 256
        out = nn.JoinTable(1, 1)({s1, s2, s3, s4})

        --model:add(nn.SpatialAveragePooling(2*opt.actionX, 2*opt.actionY))
        --model:add(nn.Reshape(nFeatures))
      else
        print("-- Critic: adding global max pooling")
        model:add(nn.SpatialMaxPooling(2*opt.actionX, 2*opt.actionY))
        model:add(nn.Reshape(nFeatures))
      end

      out = nn.Linear(nFeatures, 1024)(out)
      out = nn.LeakyReLU(true)(out)

      if opt.criticTime then
        print('Critic has timestep')
        nFeatures = nFeatures + 1
        local timestep = nn.Identity()()
        table.insert(inputs, timestep)
        local timeFeatures = nn.Linear(1, 1024)(timestep)
        timeFeatures = nn.LeakyReLU(true)(timeFeatures)
        out = nn.CAddTable()({out, timeFeatures})
      end

      out = nn.Linear(1024, 512)(out)
      out = nn.LeakyReLU(true)(out)
      
      out = nn.Linear(512, opt.max_seq_length)(out)
      --out = nn.SoftPlus()(out)

      model = nn.gModule(inputs, {out})
  elseif opt.criticVer == 'simple' or opt.criticVer == 'simple_pool' then
    print('Simple stateless critic')

    local oDims = opt.criticDim
    local shapes = {oDims, 2*oDims, 4*oDims, 8*oDims}
    if opt.dataset ~= 'mnist' and opt.dataset ~= 'mathnist' then
      table.insert(shapes, 16*oDims)
    end

    model = nn.Sequential()
    model:add(nn.JoinTable(2))

    for i = 1,#shapes do
      model:add(nn.SpatialConvolution(iDims, shapes[i], 3, 3, 1, 1, 1, 1))
      --model:add(cudnn.SpatialBatchNormalization(shapes[i]))
      model:add(nn.ReLU(true))
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      iDims = shapes[i]
    end

    if opt.criticType == 'avg' then
      model:add(nn.SpatialAveragePooling(opt.actionX, opt.actionY))
      model:add(nn.Reshape(nChannels))
      nChannels = iDims
    elseif opt.criticType == 'max' then
      model:add(nn.SpatialMaxPooling(opt.actionX, opt.actionY))
      model:add(nn.Reshape(nChannels))
      nChannels = iDims
    elseif opt.criticVer == 'softmax_cmul' then
      print("-- Critic: adding spatial softmax with CMul")
      local softmax = nn.Sequential()
                            :add(nn.LocationLogSoftMax(false))
                            :add(nn.Exp())
      model:add(nn.ConcatTable()
                      :add(softmax)
                      :add(nn.Identity()))
           :add(nn.CMulTable())
           :add(nn.Sum(3, 3))
           :add(nn.Sum(2, 2))
      nChannels = iDims
    elseif opt.criticType == 'fc' then
      nChannels = iDims*opt.actionX*opt.actionY
      model:add(nn.Reshape(nChannels))
    else
      error("Unknown -criticType " .. opt.criticType)
    end

    out = model(inputs)

    out = nn.Linear(nChannels, 1024)(out)
    out = nn.LeakyReLU(true)(out)

    if opt.criticTime then
      print('Critic has timestep')
      nFeatures = nFeatures + 1
      local timestep = nn.Identity()()
      table.insert(inputs, timestep)
      local timeFeatures = nn.Linear(1, 1024)(timestep)
      timeFeatures = nn.LeakyReLU(true)(timeFeatures)
      out = nn.CAddTable()({out, timeFeatures})
    end

    out = nn.Linear(1024, 1024):noBias()(out)
    out = nn.LeakyReLU(true)(out)

    out = nn.Linear(1024, 512):noBias()(out)
    out = nn.LeakyReLU(true)(out)

    out = nn.Linear(512, 1):noBias()(out)
    --out = nn.SoftPlus()(out)

    model = nn.gModule(inputs, {out})
  elseif opt.criticVer == 'densenet' then
    -- we're using a 121-layer net
    --stages = {6, 12, 24, 16} -- 1024
    stages = {3, 6, 12, 8}   -- 516
    --stages = {2, 4, 8, 16}
    --stages = {2, 4, 8, 12}     -- 560
  
    model = nn.Sequential()
  
    model:add(nn.JoinTable(2))
    --Initial transforms follow ResNet(224x224)
    model:add(cudnn.SpatialConvolution(iDims, nChannels, 3,3, 2,2, 3,3))
    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  
    -- Dense-Block
    for i = 1,3 do
      nChannels = addDenseBlock(model, nChannels, opt, stages[i])
      local nChannelsIn = nChannels
      addTransition(model, nChannels, math.floor(nChannels*opt.reduction), opt)
      nChannels = math.floor(nChannels*opt.reduction)
      print('Critic: Transition Down ' .. nChannelsIn .. ' -> ' .. nChannels)
    end
  
    --Dense-Block 4 and transition (7x7)
    nChannels = addDenseBlock(model, nChannels, opt, stages[4])
  
    -- Instead of transition
    model:add(nn.JoinTable(2))
    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(nn.Reshape(nChannels*opt.actionY*opt.actionX))
  
    nChannels = nChannels*opt.actionY*opt.actionX
  end

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

  --ConvInit('cudnn.SpatialConvolution')
  --ConvInit('nn.SpatialConvolution')
  --BNInit('cudnn.SpatialBatchNormalization')

  --for k,v in pairs(model:findModules('nn.Linear')) do
  --   v.bias:zero()
  --end

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
