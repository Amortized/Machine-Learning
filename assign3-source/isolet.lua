

--The IsoLet Dataset 

isolet = {};

-- The dataset has approx 6238 training + 1559 testing samples
function isolet:size() return 7797 end

function isolet:trainSize() return 6238 end

function isolet:testSize() return 1559 end

-- Each observation(feature vector) has 617 attributes 
function isolet:features() return 617 end

-- We have 26 categories
function isolet:classes() return 26 end

-- Read data from file
function isolet:readFile(filename,size)		
   -- CSV reading using simple regular expression :)
   local file = filename
   local fp = assert(io.open (file))
   local isolet_temp = {}
   local csvtable = {}
   for line in fp:lines() do
      local row = {}
      for value in line:gmatch("[-]?[%d.]*[%d]+") do
         -- note: doesn\'t work with strings that contain , values
         row[#row+1] = value
      end
      csvtable[#csvtable+1] = row
   end
   -- Generating random order
   local rorder = torch.randperm(size)
   -- iterate over rows
   for i = 1, size do
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(isolet:features())
      local output = torch.Tensor(1)
      for j = 1, isolet:features() do
         -- set entry in feature matrix
         input[j] = csvtable[i][j]
      end
      -- get class label from last column (num_features+1)
      output[1] = csvtable[i][isolet:features()+1]
      -- Shuffled dataset
      isolet_temp[rorder[i]] = {input, output}
   end
   -- Return this dataset 
   return isolet_temp
end

-- Read data from mnist_train_32x32.t7 and mnist_test_32x32.t7. Splitting
-- is automatically done.
function isolet:readFiles()
   -- Reading raw files
   --Subtable to store original data
   isolet.orig = {}
   isolet.orig.train = isolet:readFile('isolet1+2+3+4.data',isolet:trainSize())
   isolet.orig.test = isolet:readFile('isolet5.data',isolet:testSize())
end



-- Split the Isolet dataset to training and testing dataset
-- Note: mnist:readFile() must have been executed
function isolet:split(train_size, test_size)
   local train = {}
   local test = {}
   function train:size() return train_size end
   function test:size() return test_size end
   function train:features() return isolet:features() end
   function test:features() return isolet:features() end
   function train:classes() return isolet:classes() end
   function test:classes() return isolet:classes() end

   -- iterate over rows
   for i = 1,train:size() do
      -- Cloning data instead of referencing, so that the datset can be split multiple times
      train[i] = {isolet.orig.train[i][1]:clone(), isolet.orig.train[i][2]:clone()}
   end
   -- iterate over rows
   for i = 1,test:size() do
      -- Cloning data instead of referencing
      test[i] = {isolet.orig.test[i][1]:clone(), isolet.orig.test[i][2]:clone()}
   end

   -- Return the datasets
   return train, test
end

-- Normalize the dataset using training set's mean and std
-- train and test must be returned from isolet:split
function isolet:normalize(train, test)
   -- Allocate mean and variance vectors
   local mean = torch.zeros(train:features())
   local var = torch.zeros(train:features())
   -- Iterative mean computation
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]/i
   end
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
   end
   -- Get the standard deviation
   local std = torch.sqrt(var)
   -- If any std is 0, make it 1
   std:apply(function (x) if x == 0 then return 1 end end)
   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = torch.cdiv(train[i][1]-mean, std)
   end
   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = torch.cdiv(test[i][1]-mean, std)
   end

   return train, test
end

-- Add a dimension to the inputs which are constantly 1
-- This is useful to make simple linear modules without thinking about the bias
function isolet:appendOne(train, test)
   -- Sanity check. If dimensions do not match, do nothing.
   if train:features() ~= isolet:features() or test:features() ~= isolet:features() then
      return train, test
   end
   -- Redefine the features() functions
   function train:features() return isolet:features() + 1 end
   function test:features() return isolet:features() + 1 end
   -- Add dimensions
   for i = 1,train:size() do
      train[i][1] = torch.cat(train[i][1], torch.ones(1))
   end
   for i = 1, test:size() do
      test[i][1] = torch.cat(test[i][1], torch.ones(1))
   end
   -- Return them back
   return train, test
end



function isolet:getIsoletDatasets(train_size, test_size)
   if isolet.orig == nil then isolet:readFiles() end
   -- Split the datasets
   local train, test = isolet:split(train_size, test_size)
   -- Normalize the dataset
   train, test = isolet:normalize(train, test)
   -- Append one to each input
   --train, test = isolet:appendOne(train, test)
   -- return train and test datasets
   return train, test 
end

--Sanity Checker
--[[
local data_train,data_test = isolet:getIsoletDatasets(10,5)
for i = 1, table.getn(data_test) do
  print(data_test[i][1])
  print(data_test[i][2])
end]]--


