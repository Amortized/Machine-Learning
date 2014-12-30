
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")


--[[
Multi-class classification using binary classifiers implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you have to implement multOneVsAll and multOneVsOne. As an example
, part of multOneVsAll is given. These functions accept a parameter mfunc,
which is a function. Upon calling mfunc, a trainable model is returned with
whom you can run model:train(dataset) to train and return training error.
model:g(x) should give a classification, and model:l(x,y) should give the loss
on sample x y.

Of course, you can implement everything in your own way and disregard the code
here. 
--]]	

-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).

function mfunc(featureSize)
  
 
   print("Initializing a Model....")
   local model = modPrimSVM(featureSize, regL2(0.05))


   --Train this model
   function model:train(dataset)

       print("Initializing a SGD steps...")
       local trainer = trainerSGD(model, stepCons(0.0005))

       print("Training for 100 steps")
       trainer:train(dataset, 100)


       --return the train error
   end

   --This model doesn't require test function
  
   return model

end

-- Variated mfunc for One vs One
function mfunc1(featureSize, i, j)

   --Store the Pair
   local mapper = {}
   mapper[1] = i
   mapper[-1] = j

   
   print("Initializing a Model....")
   local model = modPrimSVM(featureSize, regL2(0.05))
  

   function model:map()
      return mapper
   end
 
   --Train this model
   function model:train(dataset)

       print("Initializing a SGD steps...")
       local trainer = trainerSGD(model, stepCons(0.0005))

       print("Training for 100 steps")
       trainer:train(dataset, 100)


       --return the train error
   end

   --This model doesn't require test function

   return model

end



function mfunc_xsvm()

   return xsvm.vectorized{kernel = kernLin(), C = 0.05}

end



function multOneVsAll()
   -- Create an one-vs-all trainer
   local mult = {}

   local noOfClasses = 0

   -- Transform the dataset for one versus all
   local function procOneVsAll(dataset)
      -- The data table consists of dataset:classes() datasets
      local data = {}
      -- Iterate through each dataset
      for i = 1, dataset:classes() do
	 -- Create this dataset, with size() method returning the same thing as dataset
	 data[i] = {size = dataset.size}
	 -- Modify the labels
	 for j = 1, dataset:size() do
	    -- Create entry
	    data[i][j] = {}
	    -- Copy the input
	    data[i][j][1] = dataset[j][1]
	    if dataset[j][2][1] == i then
	       -- The label same to this class i is set to 1
	       data[i][j][2] = torch.ones(1)
	    else
	       -- The label different from this class i is set to -1
	       data[i][j][2] = -torch.ones(1)
	    end
	 end
      end
      -- Return this set of datsets
      return data
   end
   -- Train models
   function mult:train(dataset)
      --Set the no of features
      local noOfFeatures = dataset:features()

      noOfClasses = dataset:classes()
      -- Define mult:classes
      mult.classes = dataset.classes
      -- Preprocess the data
      local data = procOneVsAll(dataset)
      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
	 -- Create a model
	 mult[i] = mfunc(noOfFeatures)
         --mult[i] = mfunc_xsvm()
	 -- Train the model
	 mult[i]:train(data[i])
      end
      -- Return the training error
      return mult:test(dataset)
   end
   -- Test on dataset
   function mult:test(dataset)
      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
	 -- Iterative error rate computation
	 if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error/i*(i-1)
	 else
	    error = error/i*(i-1) + 1/i
	 end
      end
      -- Return the testing error
      return error
   end
   -- The decision function
   function mult:g(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
      local maxProbability = -1e30
     
      local temp = {} 

      for i = 1, noOfClasses do
	  temp[i] = mult[i]:f(x)[1]
	  --print(temp[i])
      end

      local maxLabel
      for i = 1, noOfClasses do 
	 if (temp[i] > maxProbability ) then maxProbability = temp[i]
	 maxLabel = i
	 end
      end
      
        return torch.ones(1) * maxLabel

   end

  
   -- Return this one-vs-all trainer
   return mult
end

-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).
-- model:g(x) will determine the label of a given x.
function multOneVsOne()
   -- Create an one-vs-all trainer
   local mult = {}

   local noOfClasses = 0
   local no_of_svm = 1

   -- Transform the dataset for one versus one
   local function procOneVsOne(dataset, j, k)
      -- The data table consists of dataset:classes() datasets
      local data = {}
      count = 1
      for i = 1, dataset:size() do
        if dataset[i][2][1] == j then
	    --Create an entry
	    --data[count] = {}		
            --Copy the Input
            data[count] = {dataset[i][1]:clone(), torch.ones(1)}
            --data[count][2] = torch.ones(1)
            count = count + 1
	 end
        if dataset[i][2][1] == k then
	    --Create an entry
	    --data[count] = {}
            --Copy the Input 
            data[count] = {dataset[i][1]:clone(), -torch.ones(1)}
            --data[count][2] = -torch.ones(1)
            count = count + 1
        end

      function data:size() return count - 1 end 

     end
      -- Return this set of datsets
      return data
   end

   -- Train models
   function mult:train(dataset)
      --Set the no of features
      local noOfFeatures = dataset:features()

      noOfClasses = dataset:classes()
      
      -- Define mult:classes
      mult.classes = dataset.classes

      -- Track what has already been trained
      seen = {}
      for i = 1, dataset:classes() do
	seen[i] = {}
	for j = 1, dataset:classes() do 
	  seen[i][j] = 0
	end
      end	

      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
	for j = 1, dataset:classes() do
	    if i ~= j and seen[i][j] == 0 then 
		--Record this pair
		  seen[i][j] = 1
         	-- Create a model
	         mult[no_of_svm] = mfunc1(noOfFeatures,i,j)
		-- Get the data for this pair
		local data = procOneVsOne(dataset, i, j)
         	-- Train the model
         	mult[no_of_svm]:train(data)
		-- Increment No of SVM'S Trained
	        no_of_svm = no_of_svm + 1	
	    end
	end
      end
      -- Return the training error
      return mult:test(dataset)
   end

   -- Test on dataset
   function mult:test(dataset)
      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
         -- Iterative error rate computation
         if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
            error = error/i*(i-1)
         else
            error = error/i*(i-1) + 1/i
         end
      end
      -- Return the testing error
      return error
   end
 
      -- The decision function
   function mult:g(x)
      --Hash Map of sorts 
      myMap = {}
      for i = 1, noOfClasses do    	
	myMap[i] = 0
      end

      --Test this new feature against all Trained SVM's
      for i = 1, no_of_svm-1 do 
        res = mult[i]:g(x)[1]
	class = mult[i]:map()[res]		
	myMap[class] = myMap[class] + 1
      end

      --Return the class which has maximum votes
      value, index = torch.max(torch.Tensor(myMap),1)
      return torch.ones(1) * index[1]
   end


  --Return this all vs all trainer
  return mult
end
