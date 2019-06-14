--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CVPPP dataset
--

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt)
   local cachePath = paths.concat(opt.cache, 'kitti.t7')
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      paths.mkdir(opt.cache)

      local script = paths.dofile('kitti-gen.lua')
      script.exec(opt, cachePath)
   end

   local imageInfo = torch.load(cachePath)

   local Dataset = require('datasets/kitti')
   return Dataset(opt, imageInfo)
end

return M
