--
-- Helper function defining various ways of merging the 
-- prediction into the context mask
--

require 'nn'
require 'modules/MaskAngles'

local ContextManager = torch.class('ContextManager')

function ContextManager:__init(opt, nonbatch)
  if nonbatch then
    self.getMaskAngles = nn.MaskAnglesNonBatch(opt.imHeight, opt.imWidth)
  else
    self.getMaskAngles = nn.MaskAngles(opt.imHeight, opt.imWidth)
  end
end

function ContextManager:copyCuda()
  self.getMaskAngles = self.getMaskAngles:cuda()
end

function ContextManager:max(extMem, mask, prevMasks)
  extMem:cmax(mask)
end

function ContextManager:angles(extMem, mask, prevMasks)
  --assert(mask:sum() == mask:sum(), 'NaN detected in mask')
  --assert(prevMasks:sum() == prevMasks:sum(), 'NaN detected in prevMasks')
  --assert(extMem:sum() == extMem:sum(), 'NaN detected in extMem')
  local maskAngles = self.getMaskAngles:forward(mask)

  -- adding new mask
  extMem:sub(1, -1, 1, 1):cmul(prevMasks)
  extMem:sub(1, -1, 1, 1):add(maskAngles:sub(1, -1, 1, 1):cmul(mask))
  --
  extMem:sub(1, -1, 2, 2):cmul(prevMasks)
  extMem:sub(1, -1, 2, 2):add(maskAngles:sub(1, -1, 2, 2):cmul(mask))

  -- normalising
  prevMasks:add(mask)
  --assert(extMem:sum() == extMem:sum(), 'NaN detected in extMem after ops')
  extMem:sub(1, -1, 1, 1):cdiv(prevMasks + 1e-8)
  extMem:sub(1, -1, 2, 2):cdiv(prevMasks + 1e-8)

  local extMemMax = torch.abs(extMem):max()
  assert(extMemMax < 10.0, "ExtMem exceeded a limit: " .. extMemMax)
end
