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
function M.Resize(width, height, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(sample)
     sample.orig = sample.input:clone()
     sample.input = image.scale(sample.input, width, height, interpolation)
     return sample
   end
end

return M
