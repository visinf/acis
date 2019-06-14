require 'nn'

local analysis = require 'analysis'
local hungarian = require 'hungarian'

local BatchMaxMatch, parent = torch.class('nn.BatchMaxMatch', 'nn.Module')

function BatchMaxMatch:__init(orderby, N, noise, scoreFunc)

  self.noise = noise or false
  self.orderby = orderby or 'hung'
  print('BatchMaxMatch: ordering by ', self.orderby)

  --
  -- N maximum number of timesteps
  --
  parent.__init(self)

  function computeDice(a, b)
    local B = analysis.toBin(b[1]):typeAs(a)
    return analysis.batchDice(a[1], B)
  end

  self.outputIndices = torch.LongTensor(1, N)
  self.outputScore = torch.Tensor(1, 1)

  self.computeScore = scoreFunc or computeDice
end

function BatchMaxMatch:updateOutput(input)
  --
  -- Score Matrix [TxN]
  -- GT Masks [1xHxW]
  -- Mask [1xHxW]
  --

  local scoreMat, gtMasks, mask = unpack(input)

  local scores = self.computeScore(mask, gtMasks)

  -- adding noise (unchanged if self.noise == 0)
  scores:add(self.noise * torch.randn(scores:size()):cuda())

  local T, N = scoreMat:size(1), scores:nElement()
  local subScores = scoreMat:sub(1, -1, 1, N)
  subScores:sub(T, T):copy(scores)

  -- potential function
  local assignments
  if self.orderby == 'hung' then
    local ok
    ok, assignments = pcall(hungarian.maxCost, subScores)
    if not ok then
      print('Error in matching: ')
      print(assignments)
      print(scoreSubMtx)
      os.exit()
    end
  else
    assignments = torch.range(1, T)
  end

  assignments = assignments:long()

  self.outputIndices:zero()
  self.outputScore:zero()

  self.outputScore[1][1] = scoreMat:gather(2, assignments:view(-1, 1)):sum()
  self.outputIndices[1]:sub(1, T):copy(assignments)

  return {self.outputScore, self.outputIndices}
end

function BatchMaxMatch:updateGradInput(input, gradOutput)
  error('BatchMaxMatch does not support backprop')
end

local BatchMatchIndex, parent = torch.class('nn.BatchMatchIndex', 'nn.Sequential')

function BatchMatchIndex:__init(orderby, T, noise, scoreFunc)
  --
  -- T maximum number of timesteps
  --
  parent.__init(self)

  -- map
  self:add(nn.MapTable():add(nn.SplitTable(1)))
  self:add(nn.ZipTable())
  self:add(nn.MapTable():add(nn.BatchMaxMatch(orderby, T, noise, scoreFunc)))

  -- reduce
  self:add(nn.ZipTable())
  self:add(nn.MapTable():add(nn.JoinTable(1)))
end
