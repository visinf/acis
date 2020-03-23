local nn = require 'nn'

local function createModel(opt)
 
  local hidden = nn.Identity()()

  print("Control with " .. opt.numClasses .. " classes")

  local out = nn.Linear(opt.rnnSize, opt.featureSize)(hidden)
  out = nn.LeakyReLU(true)(out)
  out = nn.Linear(opt.featureSize, opt.numClasses)(out)
  model = nn.gModule({hidden}, {out})

  if opt.nGPU > 0 then
    model:cuda()
    if opt.cudnn == 'deterministic' then
       model:apply(function(m)
          if m.setMode then m:setMode(1,1,1) end
       end)
    end
  end
  
  return model
end

return createModel
