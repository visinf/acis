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

function M.create(opt, split)
   local cachePath = paths.concat(opt.gen, opt.dataset .. '_' .. split .. '.t7')
   print('Checking cache ' .. cachePath)
   if not paths.filep(cachePath) then
      paths.mkdir('gen')

      print('Generating cache ' .. cachePath)
      local script = paths.dofile(opt.dataset .. '-gen.lua')
      script.exec(opt, split, cachePath)
   end
   print('Loading ' .. cachePath)
   local imageInfo = torch.load(cachePath)
   print('Done')

   local Dataset = require('datasets/' .. opt.dataset)
   return Dataset(imageInfo, opt, split)
end

return M
