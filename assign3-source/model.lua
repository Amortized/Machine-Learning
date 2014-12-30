-- Import the Neural Network Package
require('nn')
require 'torch'
require 'MulPos'
require 'NegExp'
require 'RBF'


function Logistic(no_of_features, no_of_classes)
  local model = {}

  --Build the Neural Network 
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(no_of_features, no_of_classes))
  mlp:add(nn.LogSoftMax())

  --Train the Neural Network 
  function model:train(data_train)
   for i = 1, 5 do
    for i = 1, data_train:size() do
      local criterion = nn.ClassNLLCriterion()
      pred = mlp:forward(data_train[i][1])
      local err = criterion:forward(pred, data_train[i][2][1])
      mlp:zeroGradParameters()
      local t = criterion:backward(pred, data_train[i][2][1])
      mlp:backward(data_train[i][1], t)
      mlp:updateParameters(0.001)
   end
   end
    return model:test(data_train)
  end



  --Test the Model
  function model:test(data_test)
    local gold = 0
    local error = 0
    for i = 1, data_test:size() do
       value, index = torch.max(mlp:forward(data_test[i][1]) ,1)
       if(index[1] == data_test[i][2][1]) then
          error = error*(i-1)/i
          gold = gold + 1
       else
          error = (error*i-error + 1)/i
       end
    end

    --print((gold)/data_test:size())
    --print(error)      
    return error
  end

  --Return the model 
  return model

end



function neuralNetLogistic(no_of_features, no_of_classes, no_hidden_units)
  local model = {}
  
  --Build the Neural Network 
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(no_of_features, no_hidden_units))
  mlp:add(nn.Tanh())
  mlp:add(nn.Linear(no_hidden_units, no_of_classes))
  mlp:add(nn.LogSoftMax())


  --Train the Neural Network 
  function model:train(data_train)
   for i = 1, 5 do
    for i = 1, data_train:size() do
      local criterion = nn.ClassNLLCriterion()
      pred = mlp:forward(data_train[i][1])
      local err = criterion:forward(pred, data_train[i][2][1])
      mlp:zeroGradParameters()
      local t = criterion:backward(pred, data_train[i][2][1])
      mlp:backward(data_train[i][1], t)
      mlp:updateParameters(0.001)
    end
   end
    return model:test(data_train)
  end


  --Test the Model
  function model:test(data_test)
    local gold = 0
    local error = 0
    for i = 1, data_test:size() do
       value, index = torch.max(mlp:forward(data_test[i][1]) ,1)
       if(index[1] == data_test[i][2][1]) then
          error = error*(i-1)/i
          gold = gold + 1
       else
          error = (error*i-error + 1)/i
       end
    end

    --print((gold)/data_test:size())
    --print(error) 	
    return error
  end 

  --Return the model 
  return model  

end


function RBFModel(no_of_features, no_of_classes, no_hidden_units)
  local model = {}
   
  --Build the Neural Network 
  local mlp = nn.Sequential()

  mlp:add(nn.RBF(no_of_features, no_hidden_units))
  mlp:add(nn.MulPos(no_hidden_units))
  mlp:add(nn.NegExp())
  mlp:add(nn.Linear(no_hidden_units, no_of_classes))
  mlp:add(nn.LogSoftMax())

    --Train the Neural Network 
  function model:train(data_train)
   for i = 1, 5 do
    for i = 1, data_train:size() do
      local criterion = nn.ClassNLLCriterion()
      pred = mlp:forward(data_train[i][1])
      local err = criterion:forward(pred, data_train[i][2][1])
      mlp:zeroGradParameters()
      local t = criterion:backward(pred, data_train[i][2][1])
      mlp:backward(data_train[i][1], t)
      mlp:updateParameters(0.001)
    end
   end
    return model:test(data_train)
  end


  --Test the Model
  function model:test(data_test)
    local gold = 0
    local error = 0
    for i = 1, data_test:size() do
       value, index = torch.max(mlp:forward(data_test[i][1]) ,1)
       if(index[1] == data_test[i][2][1]) then
          error = error*(i-1)/i
          gold = gold + 1
       else
          error = (error*i-error + 1)/i
       end
    end

    --print((gold)/data_test:size())
    --print(error)      
    return error
  end

  --Return the model 
  return model

end

