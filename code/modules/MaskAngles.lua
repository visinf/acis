require 'nn'

-- Compute angle representation of a segment
-- similar to optical flow: x and y channels
-- point to the centroid; magnitude measures
-- the distance

local MaskAnglesNonBatch, parent = torch.class('nn.MaskAnglesNonBatch', 'nn.Module')

function MaskAnglesNonBatch:__init(height, width, cutoff)
  parent.__init(self)
  
  -- 2xhxw
  self.xyFeatures = self:createXY(height, width)
	self.cutoff = cutoff or 0.5
end

function MaskAnglesNonBatch:createXY(h, w)
  -- creates spatial features (X-Y coordinates)

	-- horisontal
  local locH = torch.Tensor(w)
                        :linspace(1, w, w)
                        :view(1, 1, w)
                        :expand(1, h, w)
	locH = locH / w

	-- vertical
  local locV = torch.Tensor(h)
                        :linspace(1, h, h)
                        :view(1, h, 1)
                        :expand(1, h, w)
	locV = locV / h

  -- 2xhxw
  return torch.cat(locH, locV, 1)
end

function MaskAnglesNonBatch:updateOutput(input)
  local _, h, w = unpack(input:size():totable())

	local xyFeatures = self.xyFeatures:typeAs(input)

  -- cutting off
  local cutoff = self.cutoff * input:max()
  local mask = torch.gt(input, cutoff):typeAs(input):cmul(input)

	-- computing the centroid
	local maskSize = mask:sum()
	local cx = torch.cmul(xyFeatures[1], mask):sum() / maskSize
	local cy = torch.cmul(xyFeatures[2], mask):sum() / maskSize

	-- computing the distance to the centroid
	local signX = torch.lt(cx - xyFeatures[1], 0)
	local signY = torch.lt(cy - xyFeatures[2], 0)

	local dx = torch.exp(torch.abs(cx - xyFeatures[1]))
	local dy = torch.exp(torch.abs(cy - xyFeatures[2]))

	dx[signX] = -dx[signX]
	dy[signY] = -dy[signY]

	dx:cmul(mask):view(1, h, w)
	dy:cmul(mask):view(1, h, w)  

	return torch.cat(dx, dy, 1):view(1, 2, h, w)
end

--
--
--

local MaskAngles, parent = torch.class('nn.MaskAngles', 'nn.Sequential')

function MaskAngles:__init(height, width, cutoff)
  parent.__init(self)

  self:add(nn.SplitTable(1))
  self:add(nn.MapTable():add(nn.MaskAnglesNonBatch(height, width, cutoff)))
	self:add(nn.JoinTable(1))
end
