--[[ 		
K-Means clustering algorithm implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The k-means algorithm should be presented here. You can implement it in any way
you want. For your convenience, a clustering object is provided at mcluster.lua

Here is how I implemented it:

kmeans(n,k) is a constructor to return an object km which will perform k-means
algorithm on data of dimension n with k clusters. The km object stores the i-th
cluster at km[i], which is an mcluster object. The km object has the following
methods:

km:g(x): the decision function to decide which cluster the vector x belongs to.
Return a scalar representing cluster index.

km:f(x): the output function to output a prototype that could replace vector x.

km:learn(x): learn the clusters using x, which is a m*n matrix representing m
data samples.
]]

dofile("mcluster.lua")

-- Create a k-means learner
-- n: dimension of data
-- k: number of clusters
function kmeans(n,k)

   clusters = {}

   -- Creates K cluster objects
   function clusters:init()
     --Create K cluster objects
     for i = 1, k do
       clusters[i] = mcluster(n);	
     end
   end
	
   --the decision function to decide which cluster the vector x belongs to.
   --Return a scalar representing cluster index.
   function clusters:g(x)
     --Go through all the cluster centroids and find out which one is the minimum
     local distance = torch.zeros(k)
     for i = 1, k do
       distance[i] = clusters[i]:eval(x);	
     end
     --Find the min
     value, index = torch.min(distance,1)
     --Return the index as a scalar
     return index[1]	
   end

   --the output function to output a prototype that could replace vector x.
   function clusters:f(x)
     return clusters[clusters:g(x)].m				
   end

   --learn the clusters using x, which is a m*n matrix representing m data samples.
   function clusters:learn(x)
      --Map to indicate which cluster each data sample belongs to	     
      data_to_cluster = torch.zeros(x:size()[1]);
      --Old Map to indicate which cluster each data sample belonged to in previous iteration
      old_data_to_cluster = torch.zeros(x:size()[1]);
      --No of successive iterations when convergence is achieved
      local convergence_counter = 0;	
      --Stores the no of samples that belong to this cluster.. Useful to allocate memory for the Tensor
      cluster_no_samples = torch.zeros(k); 		

      --Randomly select k centroids from this dataset
      y = torch.randperm(x:size()[1])	
      for i = 1, k do 
	 clusters[i]:set_m(x[y[i]]);	
      end				
			
      --Print the mean to check	
   
      while true do 
	--print(".")
	--Find the cluster
	for m = 1, x:size()[1] do
           cluster_index = clusters:g(x[m])
	   data_to_cluster[m] = cluster_index
	   cluster_no_samples[cluster_index] = cluster_no_samples[cluster_index] + 1;	  
	end

	--Set the New Means	
	for i = 1, k do
	   --Get the Samples belonging to this cluster
	   new_samples = torch.zeros(cluster_no_samples[i], n);
           count = 1;
	   for m = 1, x:size()[1] do
	      if(data_to_cluster[m] == i) then
		new_samples[count] = x[m]
		count = count + 1
	      end	
	   end	
	   --Learn new mean from these samples
	   --Assuming for now that all the samples are uniformly weighted 
	   local r = torch.ones(cluster_no_samples[i])
	   clusters[i]:learn(new_samples, r)
	end
	
	--Check for convergence
        converged = true
	for m = 1, x:size()[1] do
	   if data_to_cluster[m] ~= old_data_to_cluster[m] then
	     convergence_counter = 0		
	     converged = false
	   end
	end

	--Increment the convergence counter if convergence was achieved in this iteration

	if converged == true then 
	  convergence_counter = convergence_counter +  1
	  --If three successive converges then break
	  if convergence_counter == 3 then
	    break
	  end
	end	

	--Copy the old data 
	old_data_to_cluster = data_to_cluster:clone()
	cluster_no_samples = torch.zeros(k);

      end

   end

   --Return this object
   return clusters;
  
end

