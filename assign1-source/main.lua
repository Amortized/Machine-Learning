--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 09/22/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)
]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("mnist.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")

--Runs the Perceptron Algorithm
function Perceptron()
  print("Running Perceptron")
 
  print("Initializing Datasets...")
  local data_train, data_test = spambase:getDatasets(3000,1000)

  print("Initializing a perceptron algorithm with l2 regularization")
  local model = modPercep(data_train:features(), regL2(0.05))

  print("Initializing a batch trainer with constant step size 0.05...")
  local trainer = trainerBatch(model, stepCons(0.05))
  --local trainer = trainerBatch(model, stepHarm(0.005,1)) 
  
  print("Training for 100 batch steps...")
  local loss_train, error_train = trainer:train(data_train, 250)

  print("Testing...")
  local loss_test, error_test = trainer:test(data_test)

  print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

end

--Runs the Linear Regression trained with Direct Solution
function LinearRegressionDirectSol()
  print("Running Linear Regression with Direct Solution")

  print("Initializing Datasets...")
  local data_train, data_test = spambase:getDatasets(3000,1000)

  print("Initializing a linear regressor  with direct solution")
  local model = modLinReg(data_train:features(), regL2(0.05))

  print("Train Directly")
  model:train(data_train)

  -- This is dummy. I just use its test function
  local trainer = trainerBatch(model, stepCons(0.05))

  print("Check the Training Set")
  local loss_train, error_train = trainer:test(data_train)

  print("Testing...")
  local loss_test, error_test = trainer:test(data_test)

  print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

end

--Runs the Linear Regression trained Trained with SGD
function LinearRegressionStochastic()
  print("Running Linear Regression with Stocastic Gradient")

  print("Initializing Datasets...")
  local data_train, data_test = spambase:getDatasets(1000,1000)

  print("Initializing a linear regressor with SGD")
  local model = modLinReg(data_train:features(), regL2(0.05))

  print("Start Training for 100 steps")
  local trainer = trainerSGD(model, stepCons(0.002))
  --local trainer = trainerSGD(model, stepHarm(0.001,2))
  
  print("Training for 100 batch steps...")
  local loss_train, error_train = trainer:train(data_train, 100)
		
  print("Testing...")
  local loss_test, error_test = trainer:test(data_test)

  print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

end

--Logistic Regression with SGD
function LogisticRegressionStochastic()
  print("Running Linear Regression with Stocastic Gradient")

  print("Initializing Datasets...")
  local data_train, data_test = spambase:getDatasets(100,1000)

  print("Initializing a logistic regressor with SGD")
  local model = modLogReg(data_train:features(), regL2(0.01))

  print("Start Training for 100 steps")
  local trainer = trainerSGD(model, stepCons(0.004))
  --local trainer = trainerSGD(model, stepHarm(0.01,2))

  print("Training for 100 batch steps...")
  local loss_train, error_train = trainer:train(data_train, 125)

  print("Testing...")
  local loss_test, error_test = trainer:test(data_test)

  print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

end


--Multinomial Logistic Regression
function MultinomialLogisticRegression()
  print("Running Multinomial Logistic Regression with Stocastic Gradient")

  print("Initializing Datasets...")
  local data_train, data_test = mnist:getDatasets(100,1000)

  print("Initializing a multinomial logistic regressor with SGD")
  local model = modMulLogReg(data_train:features(), data_train:classes(), regL2(0.05))

  print("Start Training for 100 steps")
  local trainer = trainerSGD(model, stepCons(0.0005))

  print("Training for 100 batch steps...")
  local loss_train, error_train = trainer:train(data_train, 100)

  print("Testing...")
  local loss_test, error_test = trainer:test(data_test)

  print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

end




-- This is just an example
function main()



   Perceptron()
   LinearRegressionDirectSol()  
   LinearRegressionStochastic()	
   LogisticRegressionStochastic()
   MultinomialLogisticRegression()


---- EXPERIMENTS PERFORMED KINDLY IGNORE THE CODE-----
--[[   -- 1. Load spambase dataset
   print("Initializing datasets...")
   --local data_train, data_test = spambase:getDatasets(3000,1000)
   local data_train, data_test = mnist:getDatasets(1000,10)	

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05)
   print("Initializing a linear regression model with l2 regularization...")
   --local model = modLinReg(data_train:features(), regL2(0.05))
--    local model = modPercep(data_train:features(), regL1(0.05))	   
   --model:train(data_train)   

--minltr = 100000
--minlte = 100000
--minetr = 100000
--minete = 10000
--minLambda = 10000

--for lambda = 0.001, 1 , 0.001  do
   --local model = modLogReg(data_train:features(), regL2(0.05))   
   local model = modMulLogReg(data_train:features(), data_train:classes(), regL2(0.05))

   -- 3. Initialize a batch trainer with constant step size = 0.05
   print("Initializing a batch trainer with constant step size 0.05...")
   -- local trainer = trainerBatch(model, stepCons(0.05))
   local trainer = trainerSGD(model, stepCons(0.001))	

   -- 4. Perform batch training for 100 steps
   --local t1 = os.clock();
	
   print("Training for 100 batch steps...")
   local loss_train, error_train = trainer:train(data_train, 10)
   --local loss_train, error_train = trainer:test(data_train)
   
   --local dt = os.clock() - t1
   --print(dt)	

   -- 5. Perform test using the model
   print("Testing...")
   local loss_test, error_test = trainer:test(data_test)

   -- 6. Print the result
   print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)

   --if(loss_train < minltr and error_train < minetr and loss_test < minlte and error_test < minete) then
	--minltr = loss_train
	--minetr = error_train
	--minlte = loss_test
	--minete = error_test	
	--minLambda = lambda
  --end
   
--end 
--]]


end

main()
