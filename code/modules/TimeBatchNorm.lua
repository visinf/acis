local nn = require 'nn'
require 'cunn'
require 'cudnn'

local BatchNorm = nn.SpatialBatchNormalization

local TimeBatchNorm, parent = torch.class('TimeBatchNorm', 'nn.Module')

function createBunchBN(t, n)
  local bn = {}
  for i = 1,t do
	  bn_i = BatchNorm(n)
    --bn_i.gradWeight = nil
    --bn_i.gradBias = nil
    table.insert(bn, bn_i)
  end
  --collectgarbage()
  --collectgarbage()
  return bn
end

function switchMaster(master, proto)
  master.affine = proto.affine
  master.eps = proto.eps
  master.train = proto.train
  master.momentum = proto.momentum
  master.running_mean = proto.running_mean
  master.running_var = proto.running_var

  if master.affine then
     master.weight = proto.weight
     master.bias = proto.bias
     master.gradWeight = proto.gradWeight
     master.gradBias = proto.gradBias
  end
end

function TimeBatchNorm:__init(opt, nInputPlanes)
  self.timespan  = opt.max_seq_length
  self.n = nInputPlanes
  self.master = BatchNorm(nInputPlanes)
  self.bns = createBunchBN(self.timespan, nInputPlanes)
  self:setTimestep(1)
end

function TimeBatchNorm:setTimestep(t)
  assert(self.timespan >= t, "Maximum timestep exceeded")
  self.timestep = t
  switchMaster(self.master, self.bns[self.timestep])
end

function TimeBatchNorm:updateOutput(input)
  self.output = self.master:forward(input)
  return self.output
end

function TimeBatchNorm:updateGradInput(input, gradOutput)
  self.gradInput = self.master:backward(input, gradOutput)
  return self.gradInput
end
