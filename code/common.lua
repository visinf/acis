--
-- core libraries
--
require 'torch'   -- torch
require 'image'
require 'nn'      -- provides all sorts of trainable modules/layers
require 'nngraph'
require 'cunn'
require 'optim'

--
-- additional modules
--
require 'modules/KLDCriterion'
require 'modules/SamplerVAE'
require 'modules/SpatialDownSampling'
require 'modules/LocationLogSoftMax'
require 'modules/BatchMatchIndex'
require 'modules/BestScoreIndex'
require 'modules/MaskAngles'
require 'modules/DiceCriterion'
require 'modules/TimeBatchNorm'


--function getParameter(nngraph_model, name)
--    local params
--    nngraph_model:apply( function(m) if m.name==name then params = m end end)
--    return params
--end

unpack = table.unpack
