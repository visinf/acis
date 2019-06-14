-- Based on JoinTable module

require 'nn'

local SamplerVAE, parent = torch.class('nn.SamplerVAE', 'nn.Module')

function SamplerVAE:__init()
  parent.__init(self)
  self.gradInput = {}
  self.output = self.output:cuda()
end 

function SamplerVAE:updateOutput(input)
  self.output = self.output or self.output.new()
  self.output:resizeAs(input[1]):copy(input[1])
  return self.output
end
