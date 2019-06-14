local M = {}

M.getMaskTensor = function(target)
  assert(target:dim() == 2, "Expected rank 2")
  local h, w = table.unpack(target:size():totable())
  target = target:view(1, h, w)
  local mskOut = torch.IntTensor(target:size())
  local maxIdx = target:max()
  local nMsks = 0
  for ii = 1,maxIdx do
    sub = torch.eq(target, ii):int()
    if torch.sum(sub) > 0 then
      if nMsks == 0 then
        mskOut:sub(1, 1, 1, -1, 1, -1):copy(sub)
      else
        mskOut = torch.cat(mskOut, sub, 1)
      end
      nMsks = nMsks + 1
    end
  end
  return mskOut
end

M.drawMasks = function(imgIn, masks, labels, alpha, colMap, random)
  local random = random or false

  local img = imgIn:clone()
  assert(img:isContiguous() and img:dim() == 3 and masks:dim() == 3)

  local masks = masks:float()
  local ni = masks:size(1)

  local clrs = torch.Tensor(ni, 3)
  if labels then
    for l = 1,ni do
      clrs[l][1] = colMap[labels[l]][1]/255
      clrs[l][2] = colMap[labels[l]][2]/255
      clrs[l][3] = colMap[labels[l]][3]/255
    end
  else
    if random then
      colormap:setSteps(512)
      colormap:setStyle('lines')
    else
      colormap:setSteps(ni)
      colormap:setStyle('jet')
    end

    local c = 1
    local stride = math.floor(colormap.steps / ni)
    for l = 1,ni do
      clrs[l]:copy(colormap.currentMap[c])
      c = c + stride
    end
  end

  local n, h, w = masks:size(1), masks:size(2), masks:size(3)
  if not alpha then alpha=.4 end
  if not clrs then clrs=torch.rand(n,3)*.6+.4 end
  for i=1,n do
    local M = masks[i]:contiguous():data()
    local B = torch.ByteTensor(h,w):zero():contiguous():data()
    -- get boundaries B in masks M quickly
    for y=0,h-2 do for x=0,w-2 do
      local k=y*w+x
      if M[k]~=M[k+1] then B[k],B[k+1]=1,1 end
      if M[k]~=M[k+w] then B[k],B[k+w]=1,1 end
      if M[k]~=M[k+1+w] then B[k],B[k+1+w]=1,1 end
    end end
    -- softly embed masks into image and add solid boundaries
    for j=1,3 do
      local O,c,a = img[j]:data(), clrs[i][j], alpha
      for k=0,w*h-1 do if M[k]==1 then O[k]=O[k]*(1-a)+c*a end end
      for k=0,w*h-1 do if B[k]==1 then O[k]=1 end end
    end
  end

  return img
end

return M
