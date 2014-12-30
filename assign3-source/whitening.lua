
--Gets the optimal k which ensures that 99% of the variance is retained in the train set

function getOptimalK(dataset)
  --Build a matrix out of the training samples 
  local matrixDataset = torch.Tensor(dataset:size(), dataset:features())
  for i = 1,dataset:size() do
     matrixDataset[i] = dataset[i][1]:clone()
  end
  -- Get the covariance matrix 
  local sigma = torch.div(matrixDataset:transpose(1,2) * matrixDataset, dataset:size())
  -- Perform Singular Vector Decomposition on the covariance matrix to extract the eigen vectors
  local U, S, V = torch.svd(sigma)
  -- Compute the total variation in the data 
  local variation = 0 
  for i = 1, dataset:features() do
     variation = variation + S[i] 
  end
  -- Test values of k 
  local k = 1 
  while true do
     local avg_error = 0 
     for j = 1, k do 
	avg_error = avg_error + S[j]
     end   
     if(avg_error/variation >= 0.995) then 
	--99% of more variance retained
	return k
     end 
     k = k + 1	
  end  
end  


function performPCA(train_dataset, test_dataset, k)  

  
  --Build a matrix out of the training samples 
  local matrixDataset = torch.Tensor(train_dataset:size(), train_dataset:features())
  for i = 1,train_dataset:size() do 
     matrixDataset[i] = train_dataset[i][1]:clone()
  end
  
  -- Get the covariance matrix 
  local sigma = torch.div(matrixDataset:transpose(1,2) * matrixDataset, train_dataset:size())
 
  -- Perform Singular Vector Decomposition on the covariance matrix to extract the eigen vectors
  local U, S, V = torch.svd(sigma) 
      
  -- Get the first k components from U ( n * k matrix)
  local principalComponents = U:narrow(2,1,k)

  
  -- Transforming each of the n-dimensional vectors of Training Data into k-dimensional vectors
  local transformed_train_dataset = {}
  function transformed_train_dataset:size() return train_dataset:size() end
  function transformed_train_dataset:features() return k end
  function transformed_train_dataset:classes() return train_dataset:classes() end

  for i = 1, train_dataset:size() do
     local reduced = principalComponents:transpose(1,2) * train_dataset[i][1]
     --Tranpose reduced from 'k*1' to '1*k'
     transformed_train_dataset[i] = {reduced:clone(), train_dataset[i][2]:clone()}      
  end
  
  -- Transforming each of the n-dimensional vectors of Training Data into k-dimensional vectors
  local transformed_test_dataset = {}
  function transformed_test_dataset:size() return test_dataset:size() end
  function transformed_test_dataset:features() return k end
  function transformed_test_dataset:classes() return test_dataset:classes() end

  for i = 1, test_dataset:size() do
     local reduced = principalComponents:transpose(1,2) * test_dataset[i][1]
     --Tranpose reduced from 'k*1' to '1*k'
     transformed_test_dataset[i] = {reduced:clone(), test_dataset[i][2]:clone()}
  end

  --Return the transformedDataset 
  return transformed_train_dataset, transformed_test_dataset

end


function whitenDatasets(data_train, data_test, k) 
   local white_train, white_test = performPCA(data_train, data_test, k)
   return white_train, white_test
end

