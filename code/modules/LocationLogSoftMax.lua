-- applies softmax on HxW

local LocationLogSoftMax, parent = torch.class('nn.LocationLogSoftMax', 'nn.Module')

function LocationLogSoftMax:__init(inplace)
   parent.__init(self)
   inplace = inplace or false
   self.softmax = nn.LogSoftMax(inplace)
end

function LocationLogSoftMax:updateOutput(input)
  local bsz, nch, h, w = input:size(1), input:size(2), input:size(3), input:size(4)
  local inFlat = input:view(-1, h*w)
  local outFlat = self.softmax:forward(inFlat)
  return outFlat:view(bsz, nch, h, w)
end

function LocationLogSoftMax:updateGradInput(input, gradOutput)
  local bsz, nch, h, w = gradOutput:size(1), gradOutput:size(2), gradOutput:size(3), gradOutput:size(4)
  local inFlat = input:view(-1, h*w)
  local inGrad = gradOutput:view(-1, h*w)
  self.gradInput = self.softmax:backward(inFlat, inGrad):view(bsz, nch, h, w)
  return self.gradInput
end
