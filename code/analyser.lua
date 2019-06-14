analysis = require 'analysis'

local M = {}
local Analyser = torch.class('ris.Analyser', M)

function Analyser.create()
  -- The train and val loader
  return M.Analyser()
end

function Analyser:__init()
  self.size = 0
  self.criteria_stat = {}
  self.criteria_fn = {}
end

function Analyser:addMetric(name)

  if name == 'sbd' then
    self.criteria_fn[name] = analysis.sbd
  else
    error("No metric found with name " .. name)
  end

  self.criteria_stat[name] = {}
end

function Analyser:updateStat(prediction, gt_segments)
  for key,value in pairs(self.criteria_fn) do
    table.insert(self.criteria_stat[key], value(prediction, gt_segments))
  end
  self.size = self.size + 1
end

function Analyser:printStat()
  for key,value in pairs(self.criteria_stat) do
    local criteria_sum = 0
    for n = 1,self.size do
      criteria_sum = criteria_sum + value[n]
    end
    print(key .. ": ", criteria_sum / self.size)
  end
end

return M
