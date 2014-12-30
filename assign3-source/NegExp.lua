require 'nn'

local NegExp, parent = torch.class('nn.NegExp', 'nn.Module')

function NegExp:__init()
   parent.__init(self)
end

function NegExp:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:mul(-1):exp()
   return self.output 
end

function NegExp:updateGradInput(input, gradOutput) 
   self.gradInput:resizeAs(input):copy(input)
   self.gradInput:mul(-1):exp():mul(-1)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end

