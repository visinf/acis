local nn = require 'nn'
require 'cunn'
require 'nngraph'

local function createModel(opt)

  function addBlock(out, nChannelsIn, channelsOut)
    local nChannels = nChannelsIn
    for i = 1,#channelsOut do
      out = nn.SpatialConvolution(nChannels, channelsOut[i], 3,3, 1,1, 1,1)(out)
      out = nn.ReLU(true)(out)
      nChannels = channelsOut[i]
    end
    return nn.SpatialMaxPooling(2,2, 2,2)(out)
  end

  local nChannelsOut = 0
  if opt.predictAngles then
    nChannelsOut = nChannelsOut + opt.numAngles
  end

  if opt.predictFg then
    nChannelsOut = nChannelsOut + 1
  end

  local input = nn.Identity()()
  local out = nn.SpatialConvolution(3, 64, 3,3, 1,1, 100,100)(input)
  out = nn.ReLU(true)(out)

  out = addBlock(out, 64, {64})
  out = addBlock(out, 64, {128})            -- conv2_1

  local pool3 = addBlock(out, 128, {256, 256, 256})   -- conv3_1, conv3_2, conv3_3
  local pool4 = addBlock(pool3, 256, {512, 512, 512}) -- conv4_1, conv4_2, conv4_3
  out = addBlock(pool4, 512, {512, 512, 512})         -- conv5_1, conv5_2, conv5_3

  -- fc6
  out = nn.SpatialConvolution(512, 4096, 7,7, 1,1, 0,0)(out)
  out = nn.ReLU(true)(out)
  out = nn.Dropout(0.5)(out)

  -- fc7
  out = nn.SpatialConvolution(4096, 4096, 1,1, 1,1, 0,0)(out)
  out = nn.ReLU(true)(out)
  out = nn.Dropout(0.5)(out)

  -- score
  out = nn.SpatialConvolution(4096, nChannelsOut, 1,1, 1,1, 0,0)(out)

  -- upscore2
  local upscore2 = nn.SpatialFullConvolution(nChannelsOut, nChannelsOut, 4,4, 2,2)(out) -- TODO: no bias
 
  -- score_pool4
  local score_pool4 = nn.SpatialConvolution(512, nChannelsOut, 1,1, 1,1, 0,0)(pool4)
  local score_pool4c = nn.SpatialZeroPadding(-5, -5, -5, -5)(score_pool4)
  local fuse_pool4 = nn.CAddTable()({upscore2, score_pool4c})

  -- score_pool3
  local upscore_pool4 = nn.SpatialFullConvolution(nChannelsOut, nChannelsOut, 4,4, 2,2)(fuse_pool4) -- TODO: no bias
  local score_pool3 = nn.SpatialConvolution(256, nChannelsOut, 1,1, 1,1, 0,0)(pool3)
  local score_pool3c = nn.SpatialZeroPadding(-9, -9, -9, -9)(score_pool3)
  local fuse_pool3 = nn.CAddTable()({upscore_pool4, score_pool3c})

  local outputs = {}

  -- upscore8
  if opt.predictAngles then
    local upscore8Angles = nn.SpatialFullConvolution(nChannelsOut, opt.numAngles, 16,16, 8,8)(fuse_pool3) -- TODO: no bias
    local scoresAngles = nn.SpatialZeroPadding(-28, -28, -28, -28)(upscore8Angles)
    table.insert(outputs, scoresAngles)
  end

  if opt.predictFg then
    local upscore8Fg = nn.SpatialFullConvolution(nChannelsOut, 1, 16,16, 8,8)(fuse_pool3) -- TODO: no bias
    local scoresFg = nn.SpatialZeroPadding(-28, -28, -28, -28)(upscore8Fg)
    scoresFg = nn.Sigmoid()(scoresFg)
    table.insert(outputs, scoresFg)
  end

  local model = nn.gModule({input}, outputs)

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
