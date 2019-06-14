--
--  Multi-threaded data loader
--

local ffi = require 'ffi'
local datasets = require 'datasets/init'

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('net.DataLoader', M)

function DataLoader.create(opt)
  local dataset = datasets.create(opt)
  return M.DataLoader(dataset, opt)
end

function DataLoader:__init(dataset, opt)

   local function init()
     require('datasets/kitti')
   end

   local function main(idx)
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return _G.dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = 1
   self.opt = opt
   self.dataset = dataset
end

function DataLoader:size()
  return self.__size
end

function DataLoader:run()
  local threads = self.threads
  local size, batchSize = self.__size, self.batchSize

  local idx, sample = 1, nil
  local function enqueue()
     while idx <= size and threads:acceptsjob() do
        local indices = torch.LongTensor({idx})
        threads:addjob(
           function(indices)
              local sz = indices:size(1)
              assert(sz == 1, 'Error / Only batch of size 1 is supported')
              local index, inputs, paths, origs
              for i, idx in ipairs(indices:totable()) do
                 local sample_orig = _G.dataset:get(idx)
                 local sample = _G.preprocess(sample_orig)

                 if not inputs then
                    local imageSize = sample.input:size():totable()
                    inputs = torch.FloatTensor(sz, table.unpack(imageSize))
                    index = {}
                    paths = {}
                    origs = {}
                 end

                 inputs[i]:copy(sample.input)
                 table.insert(index, idx)
                 table.insert(paths, sample.path)
                 table.insert(origs, sample.orig)
              end
              local out = {}
              out.index = index[1]
              out.input = inputs
              out.orig = origs[1]
              out.path = paths[1]
              collectgarbage()
              return out
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
     return n, sample
  end
  
  return loop
end

return M.DataLoader
