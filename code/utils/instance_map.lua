
color_wheel = torch.FloatTensor({{255, 17, 0},
                                 {255, 137, 0},
                                 {230, 255, 0},
                                 {34, 255, 0},
                                 {0, 255, 213},
                                 {0, 154, 255},
                                 {9, 0, 255},
                                 {255, 0, 255}}):view(8, 3) / 255.0

function build_orientation_img(x)
    --[[
    Args:
        x: [8, H, W]
    ]]--
    local C, H, W = x:size(1), x:size(2), x:size(3)
    local y, i = torch.max(x, 1)
    local iv = color_wheel:index(1, i:long():view(-1))
    iv = iv:view(H, W, 3)
    local out = torch.cmul(iv, y:view(H, W, 1):expandAs(iv))
    return out:transpose(2,3):transpose(1,2)
end

function split_mask(mask_in)

    local imH, imW = mask_in:size(1), mask_in:size(2)
    local mask_in = mask_in:view(1, imH, imW)
    local mskOut = torch.IntTensor(mask_in:size()):zero()
    local maxIdx = mask_in:max()
    local nMsks = 0
    for ii = 1, maxIdx do
      sub = torch.eq(mask_in, ii):int()
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

function get_orientation(y, num_classes, encoding)
  local eps = 1e-8
  local C = num_classes or 8
  local encoding = encoding or 'one_hot'

  if y:dim() == 2 then
    -- converting
    y = split_mask(y)
    T, H, W = y:size(1), y:size(2), y:size(3)
    y = y:view(1, T, H, W):float()
  end

  --[[
  Args:
      y: [B, T, H, W]
  ]]--
  local B, T, H, W = y:size(1), y:size(2), y:size(3), y:size(4)
  -- [H, 1]
  local idx_y = torch.range(0, H-1):view(-1, 1):expand(H, W) 
  -- [1, W]
  local idx_x = torch.range(0, W-1):view(1, -1):expand(H, W)
  -- [H, W, 2]
  local idx_map = torch.FloatTensor(H, W, 2)
  idx_map[{{}, {}, 1}]:copy(idx_y)
  idx_map[{{}, {}, 2}]:copy(idx_x)
  -- [1, 1, H, W, 2]
  idx_map = idx_map:view(1, 1, H, W, 2):expand(B, T, H, W, 2)
  -- [B, T, H, W, 2]
  local y2 = y:view(B, T, H, W, 1)
  local y22 = y2:expand(B, T, H, W, 2)
  -- [B, T, H, W, 2]
  local y_map = torch.cmul(idx_map, y22)
  -- [B, T, 1]
  local y_sum = y:view(B, T, -1):sum(3):view(B, T, 1, 1, 1):expand(B, T, 1, 1, 2) + eps
  -- [B, T, 1, 1, 2]
  local centroids = torch.cdiv(y_map:sum(3):sum(4), y_sum)
  -- Orientation vector
  -- [B, T, H, W, 2]
  local ovec = torch.cmul((y_map - centroids:expandAs(y_map)), y22)
  -- Normalize orientation [B, T, H, W, 2]
  ovec = torch.cdiv(ovec + eps, torch.cmul(ovec, ovec):sum(5):sqrt():expandAs(ovec) + eps)
  -- [B, T, H, W]
  local angle = torch.asin(ovec[{{}, {}, {}, {}, 1}])
  local ypos = torch.gt(ovec[{{}, {}, {}, {}, 1}], 0):float()
  local xpos = torch.gt(ovec[{{}, {}, {}, {}, 2}], 0):float()
  -- [B, T, H, W]
  local angle = torch.cmul(angle, torch.cmul(xpos, ypos)) + 
                torch.cmul(math.pi - angle, torch.cmul(1 - xpos, ypos)) + 
                torch.cmul(angle, torch.cmul(xpos, 1 - ypos)) + 
                torch.cmul(-math.pi - angle, torch.cmul(1 - xpos, 1 - ypos)) + math.pi / 8
  -- [B, T, H, W]
  local angle_class = (angle + math.pi) * C / 2 / math.pi
  angle_class = torch.remainder(torch.floor(angle_class), C)
  if encoding == 'one_hot' then
      angle_class = angle_class:view(B, T, H, W, 1):expand(B, T, H, W, C)
      local clazz = torch.range(0, C - 1):float():view(1, 1, 1, 1, C):expand(B, T, H, W, C)
      local angle_one_hot = torch.eq(angle_class, clazz):float()
      -- [B, H, W, C]
      angle_one_hot:cmul(y2:expandAs(angle_one_hot))
      angle_one_hot = angle_one_hot:max(2):view(B, H, W, C)

      -- [B, C, H, W]
      angle_one_hot = angle_one_hot:transpose(3, 4):transpose(2, 3)
      return angle_one_hot:contiguous()
  elseif encoding == 'class' then
      -- [B, H, W]
      return angle_class:cmul(y):max(2):squeeze() + 1
  else
      error(string.format('Unknown encoding type: %s\n', encoding))
  end
end


function drawMasks(imgIn, masks, labels, alpha, colMap, random)
  local random = random or false

  local img = imgIn:clone()
  assert(img:isContiguous() and img:dim()==3)

  local masks = masks:float()
  local ni = masks:size(1)

  local clrs = torch.Tensor(ni, 3)
  if labels then
    assert(ni == #labels)
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
      colormap:setStyle('parula')
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
      for k=0,w*h-1 do if B[k]==1 then O[k]=c end end
    end
  end

  return img
end
