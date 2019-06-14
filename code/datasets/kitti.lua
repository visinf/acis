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
local KITTIDataset = torch.class('resnet.KITTIDataset', M)

function KITTIDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo
   self.opt = opt
   self.dir = paths.concat(opt.data, split)
   self.dir_annot = paths.concat(opt.data, 'AnnotMasksAll')
   self.split = split
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function KITTIDataset:get(i)
  local iPath = ffi.string(self.imageInfo.imagePath[i]:data())
  local mPath = ffi.string(self.imageInfo.maskPath[i]:data())

  local img = self:_loadImage(paths.concat(self.dir, iPath))

  local msk, data
  local labels = {}
  if self.opt.mode ~= 'preproc' then
    local pPath = string.gsub(iPath, "_rgb.png", "_preproc.t7")
    local fullPath = paths.concat(self.dir, pPath)

    if paths.filep(fullPath) then
      data = torch.load(fullPath)
    else
      print('Warning: ' .. fullPath .. ' doesn\'t exists!')
    end

    local tPath = string.gsub(iPath, "_rgb.png", "_labels.t7")
    local fullPath = paths.concat(self.dir, tPath)
    if paths.filep(fullPath) then
      msk = torch.load(fullPath)

      mskFg, msk = torch.max(msk, 1)
      msk = msk:int():cmul(mskFg)

      local nInstances = msk:max()


      local ni = msk:size(1)
      for i = 1,ni do table.insert(labels, 1) end
    else
      print('Warning: ' .. fullPath .. ' doesn\'t exists!')
    end

  else
    -- check if the human annotation exists:
    local fn = string.gsub(paths.basename(mPath), "_labels", "")
    local annotFile = paths.concat(self.dir_annot, fn)
    if paths.filep(annotFile) then
      print('Found ' .. annotFile)
      msk = self:_loadImage(annotFile, 1, 'byte'):squeeze()
    else
      msk = self:_loadImage(paths.concat(self.dir, mPath), 1, 'byte')
    end

  end

  return {
     input = img,
     target = msk,
     preproc = data,
     path = iPath,
     labels = labels
  }
end

function KITTIDataset:saveData(idx, data, sample)
  local data_augm = paths.concat(self.dir, 'augm')

  local dPath = paths.concat(data_augm, string.format("%06d_preproc.t7", idx))
  local iPath = paths.concat(data_augm, string.format("%06d_rgb.png", idx))
  local tPath = paths.concat(data_augm, string.format("%06d_labels.t7", idx))
  local tRGBPath = paths.concat(data_augm, string.format("%06d_labels.png", idx))

  if paths.filep(dPath) then
    print('Warning: ' .. dPath .. ' already exists!')
  else
    local inputNorm = self:renormalise(sample.input[1])
    image.save(iPath, inputNorm)

    torch.save(dPath, data)
    torch.save(tPath, sample.target[1])

    local rgbMask = self:getRGBMask(sample.target[1])
    image.save(tRGBPath, rgbMask)
  end
end

function KITTIDataset:getRGBMask(seqMask, conf)
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

-- local pca = {
--    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
--    eigvec = torch.Tensor{
--       { -0.5675,  0.7192,  0.4009 },
--       { -0.5808, -0.0045, -0.8140 },
--       { -0.5836, -0.6948,  0.4203 },
--    },
-- }

function KITTIDataset:renormalise(img)
  img = img:clone()
  for i=1,3 do
     img[i]:mul(meanstd.std[i])
     img[i]:add(meanstd.mean[i])
  end
  return img
end

function KITTIDataset:preprocess()
  if self.opt.mode == 'preproc' then
    if self.split == 'train' then
      --return function(a, b) return a, b end
      return t.Compose{--t.Resize(self.opt.xSize, self.opt.ySize),
											 t.RandomCrop(self.opt.ySize, self.opt.xSize),
                       t.ColorNormalize(meanstd),
                       t.HorizontalFlip(0.5),
                       t.PrepareKITTIMask(opt)}
   else
      --return function(a, b) return a, b end
      return t.Compose{t.Resize(self.opt.xSize, self.opt.ySize),
                       t.ColorNormalize(meanstd),
                       t.PrepareKITTIMask(opt)}
    end
  else
    return t.Compose{t.ColorNormalize(meanstd)}
  end
end

return M.KITTIDataset
