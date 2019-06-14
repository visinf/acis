--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script is adopted from imagenet-gen.lua and computes list of KITTI filenames
--
--  This generates a file gen/cvppp.t7 which contains the list of all
--  training and validation images and their segmentation masks.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findImages(dir)
   local imagePath = torch.CharTensor()
   local maskPath = torch.CharTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -maxdepth 3 -iname "*_rgb.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*_rgb.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. "/*" .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local maskPaths = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local path = string.gsub(line, dir, ""):gsub("^/", "")
      local filename_label = string.gsub(path, "_rgb", "_labels") -- TOFIX: change to '_labels' for preprocessing

      local label_path = paths.concat(dir, filename_label)
      assert(paths.filep(label_path), 'file does not exist: ' .. label_path)

      table.insert(imagePaths, path)
      table.insert(maskPaths, filename_label)

      maxLength = math.max(maxLength, #filename_label + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   print(" | found " .. nImages .. " image-label pairs")
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   local maskPath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
      ffi.copy(maskPath[i]:data(), maskPaths[i])
   end

   return imagePath, maskPath
end

function M.exec(opt, split, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, split)
   assert(paths.dirp(trainDir), split .. ' directory not found: ' .. trainDir)

   print(" | finding all training images")
   local trainImagePath, trainMaskPath = findImages(trainDir)

   local info = {
      basedir = opt.data,
      imagePath = trainImagePath,
      maskPath = trainMaskPath,
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
