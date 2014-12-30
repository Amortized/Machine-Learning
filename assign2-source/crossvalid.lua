
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")

--[[				

Cross-validation implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 10/08/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement a cross-validation mechanism that will be
used to perform multiple training on a model. You can implement it in anyway
you want, but the following is how I implemented it:

crossvalid(mfunc, k, dataset)
in which mfunc is a callable that creates a model with train and test methods,
k is the number of folds, dataset is the dataset to be used. dataset must
be randomized (our spambase and mnist datasets automatically do so when calling
getDatasets()).

The returned parameters are models, errors_train and erros_test.
				
models is atable in which models[i] should store the ith model returned by
calling mfunc.									

errors_train is a torch tensor of size k indicating training errors returned by
model[i]:train(dataset).								
			
errors_test is a torch tensor of size k indicating testing errors returned	
by model[i]:test(dataset) after training it.					
													
--]]						
			
-- How I implemented cross validation:
-- k: number of folds;
-- mfunc: a callable that creates a model with train() and test() methods.
-- model.train(dataset) should train a model and return the training error
-- model.test(dataset) should return the testing error
-- The return list is: models, errors_train, errors_test where
-- models is a table in which models[k] indicates the kth one returned by mfunc
-- errors_train is a vector of size k indicating the training errors
-- errors_test is a vector of size k indicating the cross-validation errors


-- Callable for Primal SVM
function mfuncPrimal(featureSize)
  
   print("Initializing a Model....")
   local model = modPrimSVM(featureSize, regL2(0.05))
  
   print("Initializing a SGD steps...")
   local trainer = trainerSGD(model, stepCons(0.05))
 
   --Tests on Train Errors
   function model:train(dataset)
         
       print("Training for 200 steps")
       local loss_train, error_train = trainer:train(dataset, 100)
 
       --return the train error
       return error_train
   end


   --Tests on Cross Validation Errors
   function model:test(dataset)
       print("Testing on a validation set...")    
       local loss_test, error_test = trainer:test(dataset)
       
       --return the test error
       return error_test		
   end

   return model

end

-- Callable for Kernel SVM
function mfuncKernel(C, d)
   -- 1. Initialize a dual SVM with poly kernel, and C = 0.05.
   print("Initializing a Poly kernel SVM with C ="..C.." D ="..d)
   --local model = xsvm.vectorized{kernel = kernPoly(1,2), C=0.05}
   local model = xsvm.vectorized{kernel = kernPoly(1,d), C=C}


   function model:trainer(dataset)
	-- 2. Train the kernel SVM
        print("Training the kernel SVM...")
	
	local error_train = model:train(dataset)

	return error_train
   end

   function model:tester(dataset)
	--3. Test the Kernel SVM
	print("Testing the kernel SVM...")
        local error_test = model:test(dataset)

	return error_test
   end

   return model

end


function crossvalidPrimal(k, dataset)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");


   local models = {}
   local error_train = torch.Tensor(k)
   local error_test  = torch.Tensor(k)

   local noOfFeatures = dataset:features()

   local location = 0
   local chunkSize = dataset:size()/k
   local loc = 1

for i = 1, k do

       --Declare variables      
       local train = {}
       local test = {}

       local tempDataset = {}
       for i = 1,dataset:size() do
          tempDataset  = { dataset[i][1],  dataset[i][2]}
       end
       --Get the test set  
       for j = 1, chunkSize do
         test[j] = {dataset[location + j][1]:clone(), dataset[location+j][2]:clone()}
       end

       --Get the train set
       loc = 1
       for j = 1, dataset:size() do
         if not (j >= (location+1) and j <= (location + chunkSize)) then
           --If not in the test set take it in the train set
           train[loc] = {dataset[j][1]:clone(), dataset[j][2]:clone()}
           loc = loc + 1
         end
       end

       --Set the new location
       location = location + chunkSize

       --Setting the necessary size function
       function train:size() return (dataset:size() - chunkSize) end
       function test:size() return chunkSize end


       --Get New Model and Errors
       local model = mfuncPrimal(noOfFeatures)
    
       local er_tr = model:train(train)
       local er_te = model:test(test)


       models[i] = model
       error_train[i] = er_tr
       error_test[i] = er_te
       --print ("Training Error = "..error_train[i].."Test Error = "..error_test[i])

end
       return models, error_train, error_test

end



function crossvalidKernel(k, dataset, C, d)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
  

   local models = {}
   local error_train = torch.Tensor(k)
   local error_test  = torch.Tensor(k)   
   local support_vectors = torch.Tensor(k)
   local mar_support_vectors = torch.Tensor(k)
   
   local noOfFeatures = dataset:features()

   local location = 0
   local chunkSize = dataset:size()/k   
   local loc = 1

for i = 1, k do

       --Declare variables	
       local train = {}
       local test = {}

       local tempDataset = {}
       for i = 1,dataset:size() do
          tempDataset  = { dataset[i][1],  dataset[i][2]}
       end
      

       --Get the test set  
       for j = 1, chunkSize do 
	 test[j] = {dataset[location + j][1]:clone(), dataset[location+j][2]:clone()}
       end

       --Get the train set
       loc = 1
       for j = 1, dataset:size() do
	 if not (j >= (location+1) and j <= (location + chunkSize)) then
	   --If not in the test set take it in the train set
	   train[loc] = {dataset[j][1]:clone(), dataset[j][2]:clone()}
	   loc = loc + 1
	 end
       end	

       --Set the new location
       location = location + chunkSize 

       --Setting the necessary size function
       function train:size() return (dataset:size() - chunkSize) end
       function test:size() return chunkSize end


       --Get New Model and Errors
       local model = mfuncKernel(C,d)
       local er_tr = model:trainer(train)
       local er_te = model:tester(test)
       local sp_vs = model:nsv()
       local mar_sp_vs = model:mnsv()

       models[i] = model
       error_train[i] = er_tr
       error_test[i] = er_te
       support_vectors[i] = sp_vs
       mar_support_vectors[i] = mar_sp_vs
       print ("Training Error = "..error_train[i].."Test Error = "..error_test[i].."SP"..support_vectors[i].."MSV"..mar_support_vectors[i])
	

end
       return models, error_train, error_test, support_vectors, mar_support_vectors

end


