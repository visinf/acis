require 'colormap'
require 'image'

function getRGBMask(seqMask, conf)
  local conf = conf or 0
  local nInstances = seqMask:size(1)
  local gt_mask = torch.zeros(seqMask:size(2), seqMask:size(3))
  for i = 1, nInstances do
    local mask_nz = torch.eq(gt_mask, 0):float()
    local mask_i = torch.cmul(mask_nz, torch.gt(seqMask[i], conf):float())
    gt_mask:add(torch.mul(mask_i, i):div(nInstances))
  end
  local bkg_mask = torch.gt(gt_mask, 0):double()
  local gt_mask_rgb = colormap:convert(gt_mask)
  return torch.cmul(gt_mask_rgb, bkg_mask:repeatTensor(3, 1, 1))
end

function makeVideoSequence(dir, scene, seq, result, alpha, cutoff)
  local scene = scene:clone()
  local alpha = alpha or 0.5

  local mem = scene[1]:clone():zero()

  function blend(scene, alpha, overlay)
    assert(scene:size(1) == 3, 'Unexpected image shape')
    local out = scene:clone()
    local overlay = overlay or scene[1]:clone():zero()
    for i = 1,3 do 
      out[i] = alpha*out[i] + (1 - alpha)*torch.cmul(overlay, out[i])
      out[i] = torch.cmul(mem, scene[i]) + torch.cmul(1 - mem, out[i]) 
    end
    return out
  end

  function applyMask(image1, image2, mask)
    assert(image1:size(1) == 3, 'Unexpected image shape')
    assert(image2:size(1) == 3, 'Unexpected image shape')

    for i = 1,3 do
      image1[i] = torch.cmul(1 - mask, image1[i]) + torch.cmul(mask, image2[i])
    end
    return out
  end

  function fade(duration, index, reverse)
    local nsteps = 24*duration
    local stepSize = (1 - alpha) / (nsteps - 1)
    local from, to, inc = 1, nsteps, 1
    if reverse then from, to, inc = nsteps, 1, -1 end
    for i = from,to, inc do
      local alpha_i = 1 - stepSize*(i - 1)
      local frame = blend(scene, alpha_i)
      local seqImgName = paths.concat(dir, string.format('seq_%03d.png', index))
      image.save(seqImgName, frame)
      index = index + 1
    end
    return index
  end

  function process(index)
    for t = 1,#seq do
      local masks = seq[t]
      for tt = 1,#masks do
        local frame = blend(scene, alpha, masks[tt])
        local seqImgName = paths.concat(dir, string.format('seq_%03d.png', index))
        image.save(seqImgName, frame)
        index = index + 1
      end
      local finalMask = masks[#masks]
      mem[finalMask:gt(cutoff)] = 1
      applyMask(scene, result, finalMask)

      local frame = blend(scene, alpha, finalMask)
      local seqImgName = paths.concat(dir, string.format('seq_%03d.png', index))
      image.save(seqImgName, frame)
      index = index + 1
    end

    return index
  end

  local frameId = 0

  -- Intro
  frameId = fade(1, frameId)

  -- First prediction
  local firstMask = seq[1][1]
  applyMask(scene, result, firstMask)
  mem[firstMask:gt(cutoff)] = 1
  local frame = blend(scene, alpha, firstMask)
  local seqImgName = paths.concat(dir, string.format('seq_%03d.png', frameId))
  image.save(seqImgName, frame)
  frameId = frameId + 1
  
  -- Processing
  frameId = process(frameId)

  -- Outro
  fade(1, frameId, true)
end

function getMaskTensor(target)
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

function drawMasks(imgIn, masks, labels, alpha, colMap, random)
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
