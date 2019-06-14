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
      if sample.fg then
        sample.fg = image.scale(sample.fg, sizeX, sizeY, "simple")
      end
      return sample
   end
end

function M.CenterCrop(size)
   return function(sample)
     local w1 = math.ceil((sample.input:size(3) - size)/2)
     local h1 = math.ceil((sample.input:size(2) - size)/2)
     sample.input = image.crop(sample.input, w1, h1, w1 + size, h1 + size)
     if sample.fg then
       sample.fg = image.crop(sample.fg, w1, h1, w1 + size, h1 + size)
     end
     return sample
   end
end

return M
