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
local ffi = require 'ffi'
local t = require 'datasets/transforms'

local M = {}
local CVPPPDataset = torch.class('resnet.CVPPPDataset', M)

function CVPPPDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo
   self.opt = opt
   self.dir = paths.concat(opt.data, split)
   self.split = split
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CVPPPDataset:get(i)
  local iPath = ffi.string(self.imageInfo.imagePath[i]:data())
  local mPath = ffi.string(self.imageInfo.maskPath[i]:data())

  local img = self:_loadImage(paths.concat(self.dir, iPath))
  local msk = self:_loadImage(paths.concat(self.dir, mPath), 1, 'byte')

  local data, lbl
  if self.opt.mode ~= 'preproc' then
    local pPath = string.gsub(iPath, "_rgb.png", "_preproc.t7")
    local fullPath = paths.concat(self.dir, pPath)

    if paths.filep(fullPath) then
      data = torch.load(fullPath)
    else
      print('Warning: ' .. fullPath .. ' doesn\'t exists!')
    end

    local tPath = string.gsub(iPath, "_rgb.png", "_label.t7")
    local fullPath = paths.concat(self.dir, tPath)
    if paths.filep(fullPath) then
      msk = torch.load(fullPath)
      mskFg, msk = torch.max(msk, 1)
      msk = msk:int():cmul(mskFg)

      local nInstances = msk:max()
      lbl = {}
      for i = 1,nInstances do table.insert(lbl, 1) end
    else
      print('Warning: ' .. fullPath .. ' doesn\'t exists!')
    end
    msk = msk[1]
  end

  return {
     input = img,
     target = msk,
     preproc = data,
     labels = lbl,
     path = iPath
  }
end


function CVPPPDataset:saveData(idx, data, sample)
  local data_augm = paths.concat(self.dir, 'augm')
  if not paths.dirp(data_augm) then
    print("Creating directory ", data_augm)
    paths.mkdir(data_augm)
  end

  local dPath = paths.concat(data_augm, string.format("%06d_preproc.t7", idx))
  local iPath = paths.concat(data_augm, string.format("%06d_rgb.png", idx))
  local tPath = paths.concat(data_augm, string.format("%06d_label.t7", idx))
  local tRGBPath = paths.concat(data_augm, string.format("%06d_label.png", idx))

  if paths.filep(dPath) then
    print('Warning: ' .. dPath .. ' already exists!')
  else
    local inputNorm = self:renormalise(sample.input[1])
    image.save(iPath, inputNorm)

    assert(data[1]:size(1) == 1, data[1]:size(1))
    assert(data[2]:size(1) == 1, data[2]:size(1))

    torch.save(dPath, data)
    torch.save(tPath, sample.target[1])

    local rgbMask = self:getRGBMask(sample.target[1])
    image.save(tRGBPath, rgbMask)
  end
end


function CVPPPDataset:getRGBMask(seqMask, conf)
  local mask = seqMask:float()
  mask:div(mask:max())
  local bg_mask = mask:gt(0):float()
  local gt_mask_rgb = colormap:convert(mask):float()
  return torch.cmul(gt_mask_rgb, bg_mask:repeatTensor(3, 1, 1))
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

-- local pca = {
--    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
--    eigvec = torch.Tensor{
--       { -0.5675,  0.7192,  0.4009 },
--       { -0.5808, -0.0045, -0.8140 },
--       { -0.5836, -0.6948,  0.4203 },
--    },
-- }

function CVPPPDataset:renormalise(img)
  img = img:clone()
  for i=1,3 do
     img[i]:mul(meanstd.std[i])
     img[i]:add(meanstd.mean[i])
  end
  return img
end

function CVPPPDataset:preprocess()

  if self.opt.mode == 'preproc' then
   if self.split == 'train' then
      if self.opt.preproc_save then
        return t.Compose{t.Scale(self.opt.Size),
                         t.ColorNormalize(meanstd),
                         t.HorizontalFlip(1),
                         t.VerticalFlip(1),
                         t.Rotation(180),
                         t.PrepareCVPPPMask(opt),
                         t.CenterCrop(self.opt.Size)}
      else
        return t.Compose{t.Scale(self.opt.Size),
                         t.ColorNormalize(meanstd),
                         t.PrepareCVPPPMask(opt),
                         t.CenterCrop(self.opt.Size)}
      end
   elseif self.split == 'val' then
      if self.opt.preproc_save then
        return t.Compose{t.Scale(self.opt.Size),
                         t.ColorNormalize(meanstd),
                         t.HorizontalFlip(1),
                         t.VerticalFlip(1),
                         t.PrepareCVPPPMask(opt),
                         t.CenterCrop(self.opt.Size)}
      else
        return t.Compose{t.Scale(self.opt.Size),
                         t.ColorNormalize(meanstd),
                         t.PrepareCVPPPMask(opt),
                         t.CenterCrop(self.opt.Size)}
      end
   else
      error('invalid split: ' .. self.split)
   end
 else
   if self.split == 'train' then
      --return function(a, b) return a, b end
      return t.Compose{t.ColorNormalize(meanstd)}
   elseif self.split == 'val' then
      --return function(a, b) return a, b end
      return t.Compose{t.ColorNormalize(meanstd)}
   else
      error('invalid split: ' .. self.split)
   end
 end
end

return M.CVPPPDataset
