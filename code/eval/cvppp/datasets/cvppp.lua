--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CVPPP dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local CVPPPDataset = torch.class('net.CVPPPDataset', M)

function CVPPPDataset:__init(opt, imageInfo)
   self.opt = opt
   self.imageInfo = imageInfo
   self.dir = opt.dataIn
   assert(paths.dirp(self.dir), 'cvppp: directory does not exist: ' .. self.dir)
end

function CVPPPDataset:get(i)
  local iPath = ffi.string(self.imageInfo.imagePath[i]:data())
  local img = self:_loadImage(paths.concat(self.dir, iPath))

  local fgPath = string.gsub(iPath, '_rgb', '_fg')
  local fgImg
  if paths.filep(paths.concat(self.dir, fgPath)) then
    fgImg = self:_loadImage(paths.concat(self.dir, fgPath), 1)
  end

  local tPath = string.gsub(iPath, '_rgb', '_label')
  local tgImg
  if paths.filep(paths.concat(self.dir, tPath)) then
    tgImg = self:_loadImage(paths.concat(self.dir, tPath), 1, 'byte')
  end

  return {input = img, path = iPath, fg = fgImg, target = tgImg}
end


function CVPPPDataset:_loadImage(path, nCh, dType)
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

function CVPPPDataset:size()
   return self.imageInfo.imagePath:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.5189, 0.3822, 0.2067 },
   std = { 0.2101, 0.1435, 0.1051 },
}

function CVPPPDataset:renormalise(img)
  img = img:clone()
  for i=1,3 do
     img[i]:mul(meanstd.std[i])
     img[i]:add(meanstd.mean[i])
  end
  return img
end

function CVPPPDataset:preprocess()
  return t.Compose{t.Scale(self.opt.Size),
                   t.ColorNormalize(meanstd),
                   t.CenterCrop(self.opt.Size)}
end

return M.CVPPPDataset
