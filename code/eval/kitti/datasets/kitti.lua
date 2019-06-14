--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  KITTI dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local KITTIDataset = torch.class('net.KITTIDataset', M)

function KITTIDataset:__init(opt, imageInfo)
   self.opt = opt
   self.imageInfo = imageInfo
   self.dir = opt.dataIn
   assert(paths.dirp(self.dir), 'kitti: directory does not exist: ' .. self.dir)
end

function KITTIDataset:get(i)
  local iPath = ffi.string(self.imageInfo.imagePath[i]:data())
  local img = self:_loadImage(paths.concat(self.dir, iPath))
  return {input = img, path = iPath}
end


function KITTIDataset:_loadImage(path, nCh, dType)
   nCh = nCh or 3
   dType = dType or 'float'

   local ok, input = pcall(function()
      return image.load(path, nCh, dType)
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, nCh, dType)
   end

   return input
end

function KITTIDataset:size()
   return self.imageInfo.imagePath:size(1)
end

local meanstd = {
   mean = { 0.3641, 0.3835, 0.3670 },
   std = { 0.3015, 0.3072, 0.3106 },
}

function KITTIDataset:renormalise(img)
  img = img:clone()
  for i=1,3 do
     img[i]:mul(meanstd.std[i])
     img[i]:add(meanstd.mean[i])
  end
  return img
end

function KITTIDataset:preprocess()
  return t.Compose{t.Resize(self.opt.xSize, self.opt.ySize),
                   t.ColorNormalize(meanstd)}
end

return M.KITTIDataset
