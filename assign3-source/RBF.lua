local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(inputSize,outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize,inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)


   self:reset()
end

function RBF:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   for i=1,self.weight:size(1) do
      --Multiply the weights of this Neuron with a Gaussian	
      self.weight:select(1, i):apply(function()
                                        return torch.uniform(-stdv, stdv)
                                     end)
   end
end

function RBF:updateOutput(input)
   self.output:zero()

   
   for i = 1,self.output:size(1) do       	
      --Get the weight vector correspoding to this 'i'th output neuron
      local temp = self.weight:select(1,i)
      --Set the corresponding element of the output vector	
      self.output[i] = torch.sum(torch.pow(input - temp,2))
   end

   --print(self.output)
   return self.output
end

function RBF:updateGradInput(input, gradOutput)
   --If the gradInput hasn't been compute before	
   --Zero it
      													
   for i = 1,self.weight:size(2) do
     --For each Input, get the correspoding output weight vector
     local out_weights = self.weight:select(2,i)		
     --Radial was taken for above out_weights w.r.t this input element
     local input_temp = torch.ones(gradOutput:size()):mul(input[i])
     --Compute the difference and do a dot product with corresponding gradients 
     self.gradInput[i] = 2.0 * gradOutput:dot(input_temp - out_weights)
     --end
  end
       	
  --Return the gradient 	
  return self.gradInput
end

function RBF:accGradParameters(input, gradOutput, scale)
   --self:updateOutput(input)
   scale = scale or 1
   for o = 1,self.weight:size(1) do	
      --For each output go through all the inputs	
       local temp = input - self.weight:select(1,o)	
      --Multiply every element with (-2 * scale * gradOutput[o])
       self.gradWeight[o] = self.gradWeight[o] + temp:mul((-2.0)*scale*gradOutput[o])
   end

end



