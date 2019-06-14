--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'nn'

local image = require 'image'

local M = {}

function M.Compose(transforms)
   return function(sample)
      for _, transform in ipairs(transforms) do
         sample = transform(sample)
      end
      return sample
   end
end

function M.ColorNormalize(meanstd)
   return function(sample)
     local sampleOut = sample
     for i=1,3 do
        sampleOut.input[i]:add(-meanstd.mean[i])
        sampleOut.input[i]:div(meanstd.std[i])
     end
     return sampleOut
   end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(sample)
      local w, h = sample.input:size(3), sample.input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return sample
      end
      local sizeY, sizeX
      if w < h then
        sizeY = h/w * size
        sizeX = size
      else
        sizeX = w/h * size
        sizeY = size
      end
      sample.input = image.scale(sample.input, sizeX, sizeY, interpolation)
      sample.target = image.scale(sample.target, sizeX, sizeY, 'simple')
      return sample
   end
end

-- Scales the smaller edge to size
function M.ScaleCamVid(sizeY, sizeX, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(sample)
      sample.input = image.scale(sample.input, sizeX, sizeY, interpolation)
      sample.target = image.scale(sample.target, sizeX, sizeY, 'simple')
      return sample
   end
end

-- Scales the smaller edge to size
function M.Resize(width, height, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(sample)
     sample.input = image.scale(sample.input, width, height, interpolation)
     sample.target = image.scale(sample.target, width, height, 'simple')
     return sample
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(sample)
     local w1 = math.ceil((sample.input:size(3) - size)/2)
     local h1 = math.ceil((sample.input:size(2) - size)/2)
     sample.input = image.crop(sample.input, w1, h1, w1 + size, h1 + size)
     sample.target = sample.target:sub(h1+1,h1+size, w1+1,w1+size):clone()
     return sample
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(height, width)
   return function(sample)
			local input = sample.input
			local target = sample.target

      local w, h = input:size(3), input:size(2)
      if w ~= width or h ~= height then
      	local x1, y1 = torch.random(0, w - width), torch.random(0, h - height)
      	local newInput = image.crop(input, x1, y1, x1 + width, y1 + height)
      	local newTarget = image.crop(target, x1, y1, x1 + width, y1 + height)
      	assert(newInput:size(2) == height and newInput:size(3) == width,
							 newTarget:size(2) == height and newTarget:size(3) == width, 'wrong crop size')
				sample.input = newInput
				sample.target = newTarget
      end

      return sample
   end
end

function M.VerticalFlip(prob)
   return function(sample)
      if torch.uniform() < prob then
         sample.input = image.vflip(sample.input)
         sample.target = image.vflip(sample.target)
         if sample.preproc then
           local numAngles = sample.preproc[1]:size(2)
           for j = 1,numAngles do sample.preproc[1][1][j]:copy(image.vflip(sample.preproc[1][1][j])) end
           sample.preproc[2][1][1]:copy(image.vflip(sample.preproc[2][1][1]))
         end
      end
      return sample
   end
end

function M.HorizontalFlip(prob)
   return function(sample)
      if torch.uniform() < prob then
         sample.input = image.hflip(sample.input)
         sample.target = image.hflip(sample.target)
         if sample.preproc then
           local numAngles = sample.preproc[1]:size(2)
           for j = 1,numAngles do sample.preproc[1][1][j]:copy(image.hflip(sample.preproc[1][1][j])) end
           sample.preproc[2][1][1]:copy(image.hflip(sample.preproc[2][1][1]))
         end
      end
      return sample
   end
end

function M.Rotation(deg)
   return function(sample)
      if deg ~= 0 then
         angle = (torch.uniform() - 0.5) * deg * math.pi / 180
         sample.input = image.rotate(sample.input, angle, 'bilinear')
         sample.target = image.rotate(sample.target, angle, 'simple')
         if sample.preproc then
           local angleMask = sample.preproc[1][1]
           local fgMask = sample.preproc[2][1]
           local numAngles = angleMask:size(1)
           fgMask:copy(image.rotate(fgMask, angle, 'bilinear'))
           for j = 1,numAngles do angleMask[j]:copy(image.rotate(angleMask[j], angle, 'bilinear')) end
         end
      end
      return sample
   end
end

function M.PrepareCVPPPMask(opt)

  return function(sample)
    local target = sample.target
    local mskOut = torch.IntTensor(target:size()):zero()
    local maxIdx = target:max()
    local nMsks = 0
    local labels = {}
    for ii = 1, maxIdx do
      sub = torch.eq(target, ii):int()
      if torch.sum(sub) > 0 then
        nMsks = nMsks + 1
        mskOut = mskOut + nMsks*sub
        table.insert(labels, 1)
      end
    end
    sample.target = mskOut
    sample.labels = labels
    return sample
  end

end


function M.PrepareKITTIMask(opt)

  return function(sample)
    local input = sample.input
    local target = sample.target

    local imH, imW = target:size(1), target:size(2)
    local target = target:view(1, imH, imW)
    local mskOut = torch.IntTensor(target:size())
    local maxIdx = target:max()
    local nMsks = 0
    for ii = 1, maxIdx do
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

    local masks = torch.IntTensor(nMsks, target:size(2), target:size(3))
    for ii = 1, nMsks do
    	masks[ii]:copy(mskOut[ii])
    end

    sample.target = masks
    return sample
  end

end

function M.PrepareCamVid()

  function extractMask(target)

    -- getting labels
    local lbls = {}
    local masks = {}
    for l = 0,11 do
      local msk = torch.eq(target, l):int()
      if msk:sum() > 0 then
        table.insert(masks, msk)
        table.insert(lbls, l + 1)
      end
    end

    local mskOut = torch.IntTensor(#masks, target:size(2), target:size(3))
    for ii = 1,#masks do
    	mskOut[ii]:copy(masks[ii])
    end

    return mskOut, lbls
  end

  return function(sample)
    sample.target, sample.labels = extractMask(sample.target:int())
    return sample
  end

end

return M
