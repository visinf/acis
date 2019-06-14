--[[
 		Computes the Sorensen-dice coefficient of similarity given two samples. 
 		The quotient of similarity is defined as:

    		 	    Q =     2 * (X n Y)
          		     -------------------
         		      sum_i(X) + sum_i(Y)
 		where X and Y are the two samples; 
 		(X n Y) denote the intersection where the elements of X and Y are equal.

 	Author: Olalekan Ogunmolu, July 2016
 			patlekano@gmail.com
 ]]
require 'nn'

local DICECriterion, parent = torch.class('nn.DICECriterion', 'nn.Criterion')

local eps = 1

function DICECriterion:_init(weights)
	parent._init(self)
 -- TODO: ignoring the weights
end

function DICECriterion:updateOutput(input, target)

 assert(input:nElement() == target:nElement(), "input and target size mismatch")
 assert(input:dim() == 4 and target:dim(4), "Expecting 4 dimensions")

 local bsz = input:size(1)

 self.int = torch.cmul(input, target):view(bsz, -1):sum(2)
 self.unn = torch.cmul(input, input) + torch.cmul(target, target)
 self.unn = self.unn:view(bsz, -1):sum(2)

 self.output = torch.cdiv(2*self.int, self.unn + 1e-8)

 if self.sizeAverage then
   self.output:div(self.output:nElement())
 end

 return self.output:sum()
end

function DICECriterion:updateGradInput(input, target)
 	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")
	assert(input:dim() == 4 and target:dim(4), "Expecting 4 dimensions")

 	local gradInput = self.gradInput 

 	gradInput:resizeAs(input)	

	local denom = torch.cmul(self.unn, self.unn):view(-1, 1, 1, 1):expandAs(input)
	local nom1 = torch.cmul(target, self.unn:view(-1, 1, 1, 1):expandAs(target))
	local nom2 = torch.cmul(input, self.int:view(-1, 1, 1, 1):expandAs(input))

	gradInput:copy(2*torch.cdiv(nom1 - 2*nom2, denom))

  if self.sizeAverage then
    gradInput:div(target:size(1))
  end

 	return -gradInput
end
