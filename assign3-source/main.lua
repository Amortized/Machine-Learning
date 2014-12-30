dofile("isolet.lua")
dofile("whitening.lua")
dofile("model.lua")

function preprocessData(k)
  --Gets the mean normalized data and performs dimensionality reduction using PCA 
  -- Get the datasets 
  print("Preprocess Data for k="..k)
  local data_train, data_test = isolet:getIsoletDatasets(6000,1500)
  print("Original Train Feature Size = "..data_train:features())
  print("Original Test Features Size = "..data_test:features())

  -- Whiten the Datasets 

  local white_train, white_test = whitenDatasets(data_train, data_test, k)
  -- Normalize 
  -- white_train, white_test = isolet:normalize(white_train, white_test)
  -- Append Bias after PCA has been applied
  --white_train, white_test = isolet:appendOne(white_train, white_test)
  -- Print 
  print("Reduced Train Feature Size = "..white_train:features())
  print("Reduced Test Feature Size = "..white_test:features())

  return white_train, white_test
end



function preprocessOptimal()
  --Gets the mean normalized data and performs dimensionality reduction using PCA 
  -- Get the datasets 
  print("Preprocessing Data for Best NN Model")
  local data_train, data_test = isolet:getIsoletDatasets(6000,1500)
  print("Original Train Feature Size = "..data_train:features())
  print("Original Test Features Size = "..data_test:features())

  -- Choose the number of principal components
  local k = getOptimalK(data_train)
  -- Whiten the Datasets 

  print('K='..k)
  local white_train, white_test = whitenDatasets(data_train, data_test, k)
  -- Normalize 
  -- white_train, white_test = isolet:normalize(white_train, white_test)
  -- Append Bias after PCA has been applied
  --white_train, white_test = isolet:appendOne(white_train, white_test)
  -- Print 
  print("Reduced Train Feature Size = "..white_train:features())
  print("Reduced Test Feature Size = "..white_test:features())

  return white_train, white_test

end


function doLogistic(data_train, data_test)
   --[[
   --Get the mean normalized dataset
   local data_train, data_test = isolet:getIsoletDatasets(3000,1000)
   -- Append the Bias
   data_train, data_test = isolet:appendOne(data_train, data_test) ]]--
   print("Running Logistic Model")
   -- Initialize the Neural Network Model
   local model = Logistic(data_train:features(), data_train:classes())
   -- Train the model
   local train_error = model:train(data_train)
   -- Test the Model
   local test_error  = model:test(data_test)

   --print("Training Error = "..train_error)
   print("Logistic Testing Error = "..test_error)

end

function TwoLayerNNModel(data_train,data_test)
   --[[
   --Get the mean normalized dataset
   local data_train, data_test = isolet:getIsoletDatasets(3000,1000)
   -- Append the Bias
   data_train, data_test = isolet:appendOne(data_train, data_test) ]]--
   -- Initialize the Neural Network Model
  
   print("Running 2 Layer NN Model")
   h = torch.rand(4)
   h[1] = 10 h[2] = 20 h[3] = 40 h[4] = 80

   for i = 1,4 do
   local model = neuralNetLogistic(data_train:features(), data_train:classes(), h[i])
   -- Train the model
   local train_error = model:train(data_train)
   -- Test the Model
   local test_error  = model:test(data_test)

   --print("Training Error = "..train_error)
   print("2 Layer NN Testing Error = "..test_error.." Hidden Units ="..h[i]) 
   end

end

function RadialBasisModel(data_train, data_test)
   print("Running RadialBasis Model")
   --h = torch.rand(4)
   h[1] = 10 h[2] = 20 h[3] = 40 h[4] = 80
  
   for i = 1,4 do
   local model = RBFModel(data_train:features(), data_train:classes(), h[i])
   -- Train the model
   local train_error = model:train(data_train)
   -- Test the Model
   local test_error  = model:test(data_test)

   --print("Training Error = "..train_error)
   print("Radial Basis Testing Error = "..test_error.." Hidden Units = "..h[i]) 
   end

end



function bestNNModel()
  --Get the whitened Data. No of principal components such that 99.5% of the variance is retained.
  print("Running Best NN Model")

  local data_train, data_test = preprocessOptimal()

  --Do not whiten the dataset
  local model = neuralNetLogistic(data_train:features(), data_train:classes(), data_train:features() * 4)
  -- Train the model
  local train_error = model:train(data_train)
  -- Test the Model
  local test_error  = model:test(data_test)

  --print("Training Error = "..train_error)
  print("BestNN Testing Error = "..test_error)
     
end



function main()
--[[ k = torch.rand(11)
  k[1] = 1 k[2] = 2 k[3] = 4 k[4] = 8
  k[5] = 16 k[6] = 32 k[7] = 64 k[8] = 128 k[9] = 256 k[10] = 512 k[11] = 617
  for i = 5, 11 do
	  print("K = "..k[i])]]--
  	  local white_train, white_test = preprocessData(500)
	  doLogistic(white_train, white_test)
	  TwoLayerNNModel(white_train, white_test)  
	  bestNNModel() 
	  RadialBasisModel(white_train, white_test)
--- end 

end

main()

