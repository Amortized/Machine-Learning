--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 10/10/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")
dofile("kernel.lua")
dofile("crossvalid.lua")
dofile("xsvm.lua")
dofile("mult.lua")
dofile("mnist.lua")


function linearKernel()
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)

   -- 2. Initialize a dual SVM with linear kernel, and C = 0.05.
   print("Initializing a linear kernel SVM with C = 0.05...")
   local model = xsvm.vectorized{kernel = kernLin(), C = 0.05}

   -- 3. Train the kernel SVM
   print("Training the kernel SVM...")
   local error_train = model:train(data_train)

   -- 4. Testing using the kernel SVM
   print("Testing the kernel SVM...")
   local error_test = model:test(data_test)

   -- 5. Print the result
   print("Train error = "..error_train.."; Testing error = "..error_test)

end

function polyKernel()
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)

   -- 2. Initialize a dual SVM with poly kernel, and C = 0.05.
   print("Initializing a Poly kernel SVM with C = 0.05...")
   local model = xsvm.vectorized{kernel = kernPoly(1,2), C=0.05}

   -- 3. Train the kernel SVM
   print("Training the kernel SVM...")
   local error_train = model:train(data_train)

   -- 4. Testing using the kernel SVM
   print("Testing the kernel SVM...")
   local error_test = model:test(data_test)

   -- 5. Print the result
   print("Train error = "..error_train.."; Testing error = "..error_test)

end

function PrimalSVM()
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)

   -- 2. Initialize a Primal SVM 
   print("Initializing a Primal SVM with L2 reg = 0.05...")
   local model = modPrimSVM(data_train:features(), regL2(0.05))

   print("Initializing a SGD steps...")
   local trainer = trainerSGD(model, stepCons(0.005))

   print("Training for 200 steps")
   local loss_train, error_train = trainer:train(data_train, 100)

   print("Testing...")
   local loss_test, error_test = trainer:test(data_test)

   print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)
   	
end

--------- Cross Validation on Primal SVM
function CrossValidationPrimalSVM()

   local no_of_folds = 10
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(2000,1000)

   -- 2. Calling CrossValidation function
   print("Crossvalidating...")
   local models, error_train, error_validation = crossvalidPrimal(no_of_folds, data_train)

   for i = 1, no_of_folds do 
    --Pass the model to trainer
    local trainer = trainerSGD(models[i], stepCons(0.00005))

    print("Testing...")
    local loss_test, error_test = trainer:test(data_test)

    print("Training error = "..error_train[i]..", Validation error = "..error_validation[i]..", Test Error ="..error_test)
   end
 
end

---  Cross Validation on Kernel SVM
function CrossValidationKernelSVM()

   local no_of_folds = 10
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(2000,1000)

   k = -10


   local file = assert(io.open("resultsnnew.txt", "w"))


   while k <= 10 do

   C = 2 ^ k
   d = 1
   while d<=4 do
   -- 2. Calling CrossValidation function
   print("Crossvalidating...")
   local models, error_train, error_validation = crossvalidKernel(no_of_folds, data_train, C, d)

   local error_test = {}

   for i = 1, no_of_folds do
	print("Testing the kernel SVM...")
        error_test[i] = models[i]:test(data_test)

   end

   local etr = 0
   local ev = 0
   local ete = 0
  
   for i = 1, no_of_folds do 
	etr = etr + error_train[i]
	ev  = ev + error_validation[i]
	ete = ete + error_test[i]
   end

   etr = etr/no_of_folds
   ev = ev/no_of_folds
   ete = ete/no_of_folds

   file:write("C ="..tostring(C).." d ="..tostring(d).."Training error = "..etr..", Validation error = "..ev..", Test Error ="..ete.."\n")

   --Change d
   d = d + 1
   end 
   --Change C	 
   k = k + 1
   end

   file:close()

end


function MultiClassOnevsAll()
  print("Running Multinomial Logistic Regression with Stocastic Gradient")

  print("Initializing Datasets...")
  local data_train, data_test = mnist:getDatasets(2000,1000)

  local mult = multOneVsAll()

  local train_error = mult:train(data_train)


  local test_error = mult:test(data_test)

  print("Training error ="..train_error.."Test Error = "..test_error) 

end

function MultiClassOnevsOne()
  print("Running Multinomial Logistic Regression with Stocastic Gradient")

  print("Initializing Datasets...")
  local data_train, data_test = mnist:getDatasets(2000,1000)

  local mult = multOneVsOne()

  local train_error = mult:train(data_train)

  local test_error = mult:test(data_test)

  print("Training error ="..train_error.."Test Error = "..test_error)

end

function SupportVectorsPlot()
   print("Running Support Vectors")
   local no_of_folds = 10
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(2000,1000)

   local file = assert(io.open("resul12.txt", "w"))

   C = 0.25
   d = 1
   while d<=12 do
   -- 2. Calling CrossValidation function
   print("Crossvalidating...")
   local models, error_train, error_validation, sv, msv = crossvalidKernel(no_of_folds, data_train, C, d)

   local error_test = {}

   for i = 1, no_of_folds do
        print("Testing the kernel SVM...")
        error_test[i] = models[i]:test(data_test)
   end

   local etr = 0
   local ev = 0
   local ete = 0
   local avg_sv = 0
   local avg_msv = 0

   for i = 1, no_of_folds do
        etr = etr + error_train[i]
        ev  = ev + error_validation[i]
        ete = ete + error_test[i]
	avg_sv = avg_sv + sv[i]
	avg_msv = avg_msv + msv[i]
   end

   etr = etr/no_of_folds
   ev = ev/no_of_folds
   ete = ete/no_of_folds
   avg_sv = avg_sv/no_of_folds
   avg_msv = avg_msv/no_of_folds

   file:write("C ="..tostring(C).." d ="..tostring(d).."Training error = "..etr..", Validation error = "..ev..", Test Error ="..ete.."SV ="..avg_sv.."MSV="..avg_msv.."\n")

   --Change d
   d = d + 1
   end

   file:close()

end

-- An example of using xsvm
function main()
   linearKernel()  
   polyKernel()
   PrimalSVM()
  -- CrossValidationPrimalSVM()
   CrossValidationKernelSVM()
   MultiClassOnevsAll()
   MultiClassOnevsOne()
   SupportVectorsPlot()
end

main()
