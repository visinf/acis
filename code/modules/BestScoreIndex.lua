require 'nn'

local MaxDice, parent = torch.class('nn.MaxDice', 'nn.Module')

function MaxDice:__init()
  parent.__init(self)
end
    
function MaxDice:updateOutput(input)
  assert(input:dim() == 3, 'Expecting 3 dimensions')
  assert(input:size(1) == 2, 'Expecting first dimension to be 2')

  function computeDice(a, b)
    local ab = torch.dot(a, b)
    local card = a:sum() + b:sum()
    return 2 * ab / (card + 1e-8)
  end

  -- expanding location to the batch size
  local mask, gtMasks = input[1], input[2]

  local bestIndex = -1
  local bestDice = -1

  local maxIndex = gtMasks:max()
  for i = 1,maxIndex do
    local gtMask = gtMasks:eq(i):typeAs(mask)
    local dice = computeDice(mask, gtMask)
    if dice > bestDice then
      bestDice = dice
      bestIndex = i
    end
  end

  self.output = self.output or self.output.new()
  self.output = self.output:resize(1, 2)
  self.output[1][1] = bestIndex
  self.output[1][2] = bestDice

  return self.output
end

--
-- Finds the index of ground-truth
-- corresponding to the maximum score
-- (specified by a function)
--

local BestScoreIndex, parent = torch.class('nn.BestScoreIndex', 'nn.Sequential')

function BestScoreIndex:__init(scoreFunc)
  parent.__init(self)
  local scoreFunc = scoreFunc or nn.MaxDice()

  -- input[1]: prediction mask [B, 1, H, W]
  -- input[2]: indexed ground-truth masks [B, 1, H, W]
  
  self:add(nn.JoinTable(1, 3)) -- [B, 2, H, W]
  self:add(nn.SplitTable(1)) -- {[2, H, W], [2, H, W], ...} (B times)
  self:add(nn.MapTable():add(scoreFunc)) -- [I, I, I, ...] (B times)
	self:add(nn.JoinTable(1)) -- [B, 2]
end
