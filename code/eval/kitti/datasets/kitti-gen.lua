--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script is adopted from imagenet-gen.lua and computes list of CVPPP filenames
--
--  This generates a file gen/cvppp.t7 which contains the list of all images.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findImages(dir)
   local imagePath = torch.CharTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -maxdepth 5 -iname "*_rgb.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*_rgb.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. "/*" .. findOptions)

   local maxLength = -1
   local imagePaths = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local path = string.gsub(line, dir, ""):gsub("^/", "")
      table.insert(imagePaths, path)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   print("kitti-gen: found " .. nImages .. " images")
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   return imagePath
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   print("kitti-gen: finding all training images")
   local trainImagePath = findImages(opt.dataIn)

   local info = {basedir = opt.dataIn, imagePath = trainImagePath}

   print("kitti-gen: saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
