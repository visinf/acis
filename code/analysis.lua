require 'math'

local M = {}

local function _sum_spatial(x)
  local y
  if x:dim() == 3 then
    y = torch.sum(x:view(x:size(1), -1), 2):squeeze()
  elseif x:dim() == 2 then
    y = torch.sum(x:view(-1), 1):squeeze()
  end
  return y
end

local function dice(a, b)
  --[[DICE between two segmentations.
  Args:
      a: [H, W], binary mask
      b: [T, H, W], binary mask
  
  Returns:
      dice: [T]
  --]]
  local h, w = unpack(a:size():totable())
  assert(h == b:size(2) and w == b:size(3), 'Tensor size mismatch')
  
  local n = b:size(1)

  local card_a = a:sum()
  local card_b = b:view(n, -1):sum(2)

  local aa = a:view(1, h, w):expand(n, h, w)
  local card_ab = torch.cmul(aa, b):view(n, -1):sum(2)

  local card_sum = card_a + card_b
  local c = torch.eq(card_sum, 0):typeAs(card_sum)

  return torch.cdiv(2 * card_ab, card_sum + c)
end

function M.batchDice(a, b)
  return dice(a, b)
end

function M.batchBoxOracle(a, b)
  local h, w = unpack(a:size():totable())
  assert(h == b:size(2) and w == b:size(3), 'Tensor size mismatch')
  
  local n = b:size(1)

  local aa = a:view(1, h, w):expand(n, h, w)
  local card_ab = torch.cmul(aa, b):view(n, -1):sum(2)

  return 2 * card_ab:double():cdiv(a:sum() + b:view(n, -1):sum(2):double() + 1e-16)
end

function M.batchCutBoxOracle(a, b)
  local h, w = unpack(a:size():totable())
  assert(h == b:size(2) and w == b:size(3), 'Tensor size mismatch')
  
  local n = b:size(1)

  local aa = a:view(1, h, w):expand(n, h, w)
  local card_ab = torch.cmul(aa, b):view(n, -1):sum(2)

  return card_ab:double() / (a:sum() + 1e-16)
end

local function best_dice(a, b)
  --[[For each a, look for the best DICE of all b.
  
  Args:
      a: [T, H, W], binary mask
      b: [T, H, W], binary mask
  
  Returns:
      best_dice: [T]
  --]]
  
  assert(a:dim() == 3 and 
         b:dim() == 3 and 
         a:size(2) == b:size(2) and 
         a:size(3) == b:size(3), 'Tensor size mismatch')

  -- looking for the best match for each a
  local h = a:size(2)
  local w = a:size(3)


  return best_dice
end

function toBin(a)
  local h, w = unpack(a:size():totable())
  local aMax = a:max()
  local aBin = a:view(1, h, w):expand(aMax, h, w)
  local aLin = torch.linspace(1, aMax, aMax):int():view(aMax, 1, 1):expandAs(aBin)
  return aBin:eq(aLin:typeAs(aBin)):typeAs(a)
end

function M.toBin(a)
  return toBin(a)
end

function M.sbd(a, b)
  --[[Calculates symmetric best DICE. min(BestDICE(a, b), BestDICE(b, a)).
  
  Args:
      a: [H, W], index mask
      b: [H, W], index mask
  Returns:
      sbd
  --]]
  local h, w = unpack(a:size():totable())
  assert(h == b:size(1) and w == b:size(2))
  
  -- 
  -- converting to binary masks
  --
  local aBin = toBin(a)
  local bBin = toBin(b)
  
  local nA = aBin:size(1)
  local nB = bBin:size(1)
  
  local time = sys.clock()
  
  local bd_ab = torch.zeros(nA)
  local bd_ba = torch.zeros(nB)
  
  local aSum = aBin:view(nA, -1):sum(2)
  local bSum = bBin:view(nB, -1):sum(2)
  
  for n = 1,nA do
    local maskA = aBin[n]
    local maskASum = maskA:sum()
    for nn = 1,nB do
      local maskB = bBin[nn]
      local dice = 2 * torch.cmul(maskA, maskB):sum() / (aSum[n][1] + bSum[nn][1] + 1e-8)
      bd_ab[n] = math.max(bd_ab[n], dice)
      bd_ba[nn] = math.max(bd_ba[nn], dice)
    end
  end
  
  return math.min(bd_ab:mean(), bd_ba:mean())
end

-- 
-- (un)weighted coverage
--
function M.cov(a, b)
  --[[Calculates weighted and unweighted coverage

  Args:
      a: [H, W], index mask, prediction
      b: [H, W], index mask, ground-truth
  Returns:
      cov
  --]]
  local h, w = unpack(a:size():totable())
  assert(h == b:size(1) and w == b:size(2))

  -- 
  -- converting to binary masks
  --
  local aBin = toBin(a)
  local bBin = toBin(b)

  local nA = aBin:size(1)
  local nB = bBin:size(1)

  local aSum = aBin:view(nA, -1):sum(2)
  local bSum = bBin:view(nB, -1):sum(2)

  -- computing weights
  local wt = bBin:view(nB, -1):sum(2)
  wt:div(wt:sum())

  local cov = torch.zeros(nB)

  -- computing the coverage
  for n = 1,nB do
    local maskB = bBin[n]
    for nn = 1,nA do
      local maskA = aBin[nn]
      local ab = torch.dot(maskA, maskB)
      local iou = ab / (aSum[nn][1] + bSum[n][1] - ab + 1e-8)
      cov[n] = math.max(cov[n], iou)
    end
  end

  return cov:mean(), torch.cmul(cov, wt:typeAs(cov)):sum()
end


local function _spatial_dot(a, b)
  assert(a:dim() == b:dim() and a:dim() == 4)

  local N = a:size(2)
  local dot = torch.sum(torch.cmul(a, b):view(1, N, -1), 3)

  return dot:view(1, N)
end

function M.softIoU(a, b)
  local ab = torch.dot(a:view(-1), b:view(-1))
  local aa = torch.sum(a)
  local bb = torch.sum(b)
  return ab / (aa + bb - ab)
end


function M.iou(a, b, conf)
  local conf = conf or 0.5

  local a = torch.ge(a, conf):float()

  local ab = torch.dot(a:view(-1), b:view(-1))
  local aa = torch.sum(a)
  local bb = torch.sum(b)
  return ab / (aa + bb - ab)
end

function M.dice(pred, truth, conf)
  local conf = conf or 0.5

  -- [0, 1] => {0, 1}
  local p = torch.ge(pred, conf):float()
  local t = truth:float()

  return 2 * torch.dot(p:view(-1), t:view(-1)) / (p:sum() + t:sum())
end

function M.AP(pred, truth, avgs, conf)

  local conf = conf or 0.5

  -- [0, 1] => {0, 1}
  local p = torch.ge(pred, conf):float()
  local t = truth:float()

  -- precision
  local tp = torch.dot(p:view(-1), t:view(-1))
  local fp = torch.sum(p) - tp
  local pr = tp > 0 and tp / (tp + fp) or 0

  -- iou
  local pt = torch.sum(p) + torch.sum(t) - tp
  local iou = tp / pt

  local prs = torch.cumsum(pr * torch.lt(avgs, iou):float())
  local prn = torch.range(1, avgs:size(1))

  return prs:cdiv(prn):mean()
end

function M.soft_dice(mask, targets)
  if mask:dim() + 1 == targets:dim() then
    mask = mask:view(mask:size(1), 1, mask:size(2), mask:size(3))
  end
  local N = targets:size(2)
  local mask_ex = mask:expandAs(targets):contiguous()
  local isc = _spatial_dot(mask_ex, targets)
  --local mask_isc = _spatial_dot(mask_ex, mask_ex)
  --local targets_isc = _spatial_dot(targets, targets)
  local mask_isc = mask_ex:view(1, N, -1):sum(3):view(1, N)
  local targets_isc = targets:view(1, N, -1):sum(3):view(1, N)
  return torch.cdiv(2*isc, mask_isc + targets_isc)
end

function M.soft_iou(mask, targets)
  if mask:dim() + 1 == targets:dim() then
    mask = mask:view(mask:size(1), 1, mask:size(2), mask:size(3))
  end
  local N = targets:size(2)
  local mask_ex = mask:expandAs(targets):contiguous()
  local isc = _spatial_dot(mask_ex, targets)
  --local mask_isc = _spatial_dot(mask_ex, mask_ex)
  --local targets_isc = _spatial_dot(targets, targets)
  local mask_isc = mask_ex:view(1, N, -1):sum(3):view(1, N)
  local targets_isc = targets:view(1, N, -1):sum(3):view(1, N)
  return torch.cdiv(isc, mask_isc + targets_isc - isc)
end

return M
