local THNN = require 'nn.THNN'
local SpatialDownSampling, parent = torch.class('nn.SpatialDownSampling', 'nn.Module')

function SpatialDownSampling:__init(nInputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self)
  
  dW = dW or 1
  dH = dH or 1
  
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nInputPlane
  self.kW = kW
  self.kH = kH
  
  self.dW = dW
  self.dH = dH
  self.padW = padW or 0
  self.padH = padH or self.padW
  
  self.kernel = torch.Tensor(self.nOutputPlane, nInputPlane, kH, kW)

  -- bilinear kernel
  self.K = torch.Tensor(kH, kW)
  local cx = (self.kW + 1.0) / 2.0
  local cy = (self.kH + 1.0) / 2.0

  local sdy  = 2.0 * self.dW / 6.0
  local sdx  = 2.0 * self.dH / 6.0
  local vy = sdy * sdy
  local vx = sdx * sdx
  
  for i = 1,kW do
    local dx = (cx-i)*(cx-i)
    for j = 1,kH do
      local dy = (cy-j)*(cy-j)
      local val = math.exp(-0.5*(dx/vx+dy/vy)) 
      self.K[j][i] = val
    end
  end

  self.K = self.K / torch.sum(self.K)
  self:reset()

  self.gradInput = nil
end

function SpatialDownSampling:reset()
  self.kernel:zero()
  for i = 1,self.nInputPlane do
    self.kernel[i][i]:copy(self.K)
  end
end

local function backCompatibility(self)
   self.finput = self.finput or self.kernel.new()
   self.fgradInput = self.fgradInput or self.kernel.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.kernel:dim() == 2 then
      self.kernel = self.kernel:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function SpatialDownSampling:updateOutput(input)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   backCompatibility(self)
   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.kernel:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end

function SpatialDownSampling:updateGradInput(input, gradOutput)
end

function SpatialDownSampling:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function SpatialDownSampling:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end
