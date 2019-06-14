require 'nn'

local analysis = require 'analysis'
local hungarian = require 'hungarian'

local BoxMatchIoU, parent = torch.class('nn.BoxMatchIoU', 'nn.Module')

function BoxMatchIoU:__init(opt)
  --
  -- N maximum number of timesteps
  --
  parent.__init(self)

  function computeBoxOracle(a, b)
    local B = analysis.toBin(b[1]):typeAs(a)
    return analysis.batchBoxOracle(a[1], B)
  end

  function computeCutBoxOracle(a, b)
    local B = analysis.toBin(b[1]):typeAs(a)
    return analysis.batchCutBoxOracle(a[1], B)
  end

  local imageH, imageW = opt.imHeight, opt.imWidth

  self.xcoords = torch.range(1, imageW):view(1, imageW)
                                       :expand(imageH, imageW)

  self.ycoords = torch.range(1, imageH):view(imageH, 1)
                                       :expand(imageH, imageW)

  self.computeScore = computeBoxOracle
  self.computeCutScore = computeCutBoxOracle
end

function BoxMatchIoU:updateOutput(input)
  --
  -- Score Matrix [TxN]
  -- GT Masks [1xHxW]
  -- Mask [1xHxW]
  --

  function bboxToMasks(bboxes, canvas)
    local C, H, W = unpack(canvas:size():totable())
    local numBoxes = bboxes:size(1)
    for b = 1,numBoxes do
      local xc, yc, w, h = unpack(bboxes[b]:totable())
      local y0 = math.max(1, math.floor(yc - h/2))
      local y1 = math.min(H, math.ceil(yc + h/2))
      local x0 = math.max(1, math.floor(xc - w/2))
      local x1 = math.min(W, math.ceil(xc + w/2))
      canvas[1]:sub(y0, y1, x0, x1):fill(b)
    end
  end

  function maskToBbox(mask, canvas)
    local xmin, xmax = self.xcoords[mask]:min(), self.xcoords[mask]:max()
    local ymin, ymax = self.ycoords[mask]:min(), self.ycoords[mask]:max()
    canvas[1]:sub(ymin, ymax, xmin, xmax):fill(1)
    return canvas
  end

  local masks, bboxes = unpack(input)
  local c, h, w = unpack(masks:size():totable())

  local bboxMasks = torch.zeros(1, h, w):typeAs(masks)
  bboxToMasks(bboxes, bboxMasks)

  local numMasks = masks:max()
  local numBoxes = bboxMasks:max()
  local scoreMat = torch.DoubleTensor(numMasks, numBoxes)
  --local scoreMatOracle = torch.DoubleTensor(numMasks, numBoxes)

  local bboxCanvas = torch.zeros(1, h, w):typeAs(masks)
  for t = 1,numMasks do
    local mask = masks:eq(t)
    local maskBbox = maskToBbox(mask, bboxCanvas)

    local scores = self.computeScore(maskBbox, bboxMasks)
    scoreMat[t]:copy(scores)

    --local scoresOracle = self.computeCutScore(mask, bboxMasks)
    --scoreMatOracle[t]:copy(scoresOracle)

    bboxCanvas:zero()
  end

  -- potential function
  local ok, assignments = pcall(hungarian.maxCost, scoreMat)
  if not ok then
    print('Error in matching: ')
    print(assignments)
    print(scoreSubMtx)
    os.exit()
  end

  assignments = assignments:long()

  self.outputIndices = torch.LongTensor(1, numMasks):zero()
  self.outputScore = torch.Tensor(1, 1):zero()

  local assignIndices = torch.range(1, numMasks)[assignments:gt(0)]:long()
  local scoreMatSub = scoreMat:index(1, assignIndices)

  local assignmentsNonZero = assignments:index(1, assignIndices)
  local scoreMean = scoreMatSub:gather(2, assignmentsNonZero:view(-1, 1)):sum() / numMasks

  self.outputScore[1][1] = scoreMean
  self.outputIndices[1]:copy(assignments)

  return {self.outputScore, self.outputIndices}
end

function BoxMatchIoU:updateGradInput(input, gradOutput)
  error('BoxMatchIoU does not support backprop')
end

local BoxMatch, parent = torch.class('nn.BoxMatch', 'nn.Sequential')

function BoxMatch:__init(opt)
  --
  -- T maximum number of timesteps
  --
  parent.__init(self)

  -- map
  self:add(nn.ParallelTable()
                      :add(nn.SplitTable(1))
                      :add(nn.Identity()))
  self:add(nn.ZipTable())
  self:add(nn.MapTable():add(nn.BoxMatchIoU(opt)))

  -- reduce
  self:add(nn.ZipTable())
  --self:add(nn.PrintSize())
  --self:add(nn.MapTable():add(nn.JoinTable(1)))
end
