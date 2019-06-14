local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

function KLDCriterion:__init(k)
   parent.__init(self)
end

function KLDCriterion:updateOutput(input)
    --return self.output
    local mean, log_var = unpack(input)
    self.output = input 

    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, torch.pow(mean, 2))
    KLDelements:add(1)
    KLDelements:add(log_var)

    return self.output
end
