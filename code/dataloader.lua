--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local ffi = require 'ffi'
local datasets = require 'datasets/init'

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt, split)
  -- The train and val loader
  local dataset
  dataset = datasets.create(opt, split)
  return M.DataLoader(dataset, opt, split)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed

   local function init()
     _G.ffi = require 'ffi'
     require 'utils/ContextManager'
     require('datasets/' .. opt.dataset)
     if opt.mode == 'preproc' then
       require('utils/instance_map')
     end
     if not unpack then unpack = table.unpack  end
   end

   local function main(idx)
     if manualSeed ~= 0 then
       print("Setting seed " .. (manualSeed + idx))
       torch.manualSeed(manualSeed + idx)
       math.randomseed(manualSeed + idx)
     end
     torch.setnumthreads(1)
     _G.dataset = dataset
     _G.preprocess = dataset:preprocess()

     return _G.dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = opt.batch_size
   self.gen = torch.Generator()
   self.opt = opt
   torch.manualSeed(self.gen, manualSeed)
   self.dataset = dataset
end

function DataLoader:size()
  local size = math.min(opt.maxEpochSize, self.__size)
  return math.ceil(size / self.batchSize)
end

function DataLoader:run(seqLength, cxtUpdate)

  local threads = self.threads
  local size, batchSize = self.__size, self.batchSize
  local perm = torch.randperm(self.gen, size)
  local mode = self.opt.mode

  local predictAngles = self.opt.predictAngles
  local predictFg = self.opt.predictFg
  local numAngles = self.opt.numAngles
  local imageCh, imageH, imageW = self.opt.imageCh, 
                                  self.opt.imHeight,
                                  self.opt.imWidth
  local seqLength = seqLength or self.opt.seq_length

  size = math.min(opt.maxEpochSize, size)
  local maxSeqLength = self.opt.max_seq_length
  local cxtSize = self.opt.cxtSize
  local cxtUpdate = cxtUpdate
  local orderby = self.opt.orderby

  local idx, sample = 1, nil
  local function enqueue()
     while idx <= size and threads:acceptsjob() do
        -- ensure always return batch size
        local offset = batchSize - math.min(batchSize, size - idx + 1)
        idx = idx - offset
        local indices = perm:narrow(1, idx, batchSize)
        threads:addjob(
           function(indices)
              local sz = indices:size(1)
              local scene = torch.FloatTensor(sz, imageCh, imageH, imageW)
              local targets = torch.IntTensor(sz, 1, imageH, imageW)
              local maskSum = torch.FloatTensor(sz, 1, imageH, imageW):zero()      -- just a sum of all masks
              local extMem = torch.FloatTensor(sz, cxtSize, imageH, imageW):zero() -- context representation
              local offset = torch.zeros(sz, 1):long()
              local index = torch.LongTensor(sz)
              local labels = torch.IntTensor(sz, maxSeqLength):zero()
              local paths = torch.CharTensor(sz, 255)

              local ret = {}

              if predictAngles then
                ret.angles = torch.FloatTensor(sz, numAngles, imageH, imageW)
              end

              if predictFg then
                ret.fg = torch.FloatTensor(sz, 1, imageH, imageW)
              end

              for i, idx in ipairs(indices:totable()) do
                 local sample_raw = _G.dataset:get(idx)
                 local sample = _G.preprocess(sample_raw)

                 local imageSize = sample.input:size():totable()
                 local targetSize = sample.target:size():totable()
                 local h, w = unpack(targetSize)
                 assert(imageSize[2] == h and imageSize[3] == w)

                 -- # of instances is the max index in the mask
                 local nInstances = sample.target:max()

                 if predictAngles then
                   -- feeding GT 
                   if mode == 'preproc' then
                     local angles = get_orientation(sample.target, numAngles):squeeze(1)
                     ret.angles[i]:copy(angles)
                   else
                     ret.angles[i]:copy(sample.preproc[1])
                   end
                 end

                 if predictFg then
                   if mode == 'preproc' then
                     -- feeding GT
                     ret.fg[i]:copy(sample.target:gt(0):float())
                   else
                     ret.fg[i]:copy(sample.preproc[2])
                   end
                 end

                 if sample.path then
                   _G.ffi.copy(paths[i]:data(), sample.path)
                 end

                 local startFrom = 0 -- number of GT added in the context

                 -- if # of instances > seqLength update extMem with random
                 -- selection of GT masks and reorder the rest
                 if nInstances > seqLength then
                   -- selecting a random timestep to begin with
                   -- (# of instances in the extMem)
                   -- should be in [0, nInstances - seqLength]

                   if seqLength > 0 then
                     startFrom = math.random(0, nInstances - seqLength)
                   else
                     startFrom = math.max(0, nInstances - math.abs(seqLength))
                   end

                   local instanceIds = torch.randperm(nInstances)

                   -- updating extMem with startFrom instances
                   for j = 1,startFrom do
                     local idx = instanceIds[j]
                     local gtMask = torch.eq(sample.target, idx)
                     sample.target[gtMask] = 0

                     gtMask = gtMask:typeAs(maskSum)
                     cxtUpdate(extMem:sub(i, i), gtMask, maskSum:sub(i, i))
                     maskSum[i][1]:add(gtMask)
                   end

                   -- the rest are in the GT instance pool (only trainLength will be selected)
                   local labels = {}
                   local sortedIndices = torch.sort(instanceIds:sub(startFrom + 1, nInstances))
                   for j = 1,sortedIndices:nElement() do
                     local idx = sortedIndices[j]
                     table.insert(labels, sample.labels[idx])

                     local maskIdx = #labels
                     local gtMask = torch.eq(sample.target, idx)
                     sample.target[gtMask] = maskIdx
                   end

                   sample.labels = labels
                 end

                 -- let's sort: 1 - smallest, max - largest
                 if orderby == 'size' then
                   local nInstances = sample.target:max()
                   local maskSize = {}
                   for n = 1,nInstances do
                     table.insert(maskSize, sample.target:eq(n):int():sum())
                   end
                   local _, mi = torch.sort(torch.Tensor(maskSize))
                   local targetSorted = sample.target:clone():zero()
                   for n = 1,nInstances do
                     local mask = sample.target:eq(mi[n]):typeAs(targetSorted)
                     targetSorted:add(n*mask)
                   end

                   sample.target = targetSorted
                 end

                 scene[i]:copy(sample.input)
                 targets[i]:copy(sample.target)
                 labels[i]:sub(1, #sample.labels):copy(torch.IntTensor(sample.labels))
                 index[i] = idx
                 offset[i] = startFrom
              end

              ret.idx = index
              ret.input = scene
              ret.extMem = extMem
              ret.maskSum = maskSum
              ret.target = targets
              ret.labels = labels
              ret.offset = offset
              ret.paths = paths

              collectgarbage()
              return ret
           end,
           function(_sample_)
              sample = _sample_
           end,
           indices
        )
        idx = idx + batchSize
     end
  end

  local n = 0
  local function loop()
     enqueue()
     if not threads:hasjob() then
        return nil
     end
     threads:dojob()
     if threads:haserror() then
        threads:synchronize()
     end
     enqueue()
     n = n + 1
     return sample
  end

  return loop
end

return M.DataLoader
