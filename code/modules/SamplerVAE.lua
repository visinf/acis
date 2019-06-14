-- Based on JoinTable module

require 'nn'

local SamplerVAE, parent = torch.class('nn.SamplerVAE', 'nn.Module')

function SamplerVAE:__init()
  parent.__init(self)
  self.gradInput = {}
  self.train = true
  self.fixed = true
end 

function SamplerVAE:updateOutput(input)
  if self.train and not self.fixed then
    self.eps = self.eps or input[1].new()
    self.eps:resizeAs(input[1]):copy(torch.randn(input[1]:size()))
    
    self.output = self.output or self.output.new()
    self.output:resizeAs(input[2]):copy(input[2])
    self.output:mul(0.5):exp():cmul(self.eps)
    self.output:add(input[1])
  else
    self.output = self.output or self.output.new()
    self.output:resizeAs(input[1]):copy(input[1])
  end
  
   return self.output
end

function SamplerVAE:updateGradInput(input, gradOutput)
  if self.train and not self.fixed then
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
    
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput):copy(input[2])
    
    self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)
  else
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
    
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput):zero()
  end

  return self.gradInput
end
