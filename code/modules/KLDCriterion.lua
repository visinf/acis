local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

function KLDCriterion:__init(k)
   parent.__init(self)
   self.KLDK = k or 1.0
   print('--   KLD coefficient: ' .. self.KLDK)
end

function KLDCriterion:updateOutput(input)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    --local KLDelements = log_var:clone()
    --KLDelements:exp():mul(-1)
    --KLDelements:add(-1, torch.pow(mean, 2))
    --KLDelements:add(1)
    --KLDelements:add(log_var)

    --self.output = -0.5 * torch.sum(KLDelements)
 
    --return self.output
    local mean, log_var = unpack(input)
    self.output = input 

    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, torch.pow(mean, 2))
    KLDelements:add(1)
    KLDelements:add(log_var)
    self.loss = -0.5 * torch.sum(KLDelements:mean(2))
    self.avgMean = torch.mean(mean, 2):sum()
    self.avgVar = torch.mean(torch.exp(0.5*log_var), 2):sum()

    return self.output
end

function KLDCriterion:updateGradInput(input, gradOutput)
  assert(#gradOutput == 2)
  local mean, log_var = unpack(input)
  self.gradInput = {}

  if self.KLDK > 0 then
    self.gradInput[1] = self.KLDK * mean:clone() + gradOutput[1]
    self.gradInput[2] = self.KLDK * torch.exp(log_var):mul(-1):add(1):mul(-0.5) + gradOutput[2]
  else
    self.gradInput[1] = gradOutput[1]
    self.gradInput[2] = gradOutput[2]
  end

  return self.gradInput
end
