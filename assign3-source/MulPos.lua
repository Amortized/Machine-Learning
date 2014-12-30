require 'nn'

local MulPos, parent = torch.class('nn.MulPos', 'nn.Module')

function MulPos:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

 
function MulPos:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end	

   self.weight[1] = torch.uniform(-stdv, stdv);
end

function MulPos:updateOutput(input)
   self.output:copy(input);
   self.output:mul(math.exp(self.weight[1]));
   return self.output 
end

function MulPos:updateGradInput(input, gradOutput) 
   self.gradInput:zero()
   self.gradInput:add(math.exp(self.weight[1]), gradOutput)
   return self.gradInput
end

function MulPos:accGradParameters(input, gradOutput, scale) 
   scale = scale or 1

   self.gradWeight[1] = self.gradWeight[1] + scale*self.output:dot(gradOutput);
end
