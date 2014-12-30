--[[
Mixture of Gaussians Implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The mixture of gaussians algorithm should be presented here. You can implement
it in anyway you want. For your convenience, a multivariate gaussian object is
provided at gaussian.lua.

Here is how I implemented it:

mog(n,k) is a constructor to return an object m which will perform MoG
algorithm on data of dimension n with k gaussians. The m object stores the i-th
gaussian at m[i], which is a gaussian object. The m object has the following
methods:

m:g(x): The decision function which returns a vector of k elements indicating
each gaussian's likelihood

m:f(x): The output function to output a prototype that could replace vector x.

m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix
representing m data samples. p is regularization to keep each gaussian's
covariance matrices non-singular. eps is a stop criterion.
]]

dofile("gaussian.lua")
dofile("kmeans.lua")

-- Create a MoG learner
-- n: dimension of data
-- k: number of gaussians
function mog(n,k)
-- Remove the following line and add your stuff
-- print("You have to define this function by yourself!");
  
   gaussians = {}

   --A vector of size k, where weights[i] indicates the weight of class 'i'
   weights = torch.zeros(k)
     	 
   --Create Gaussian objects 
   --Initialize the mean and variance of Gausians and Wj
   --p is the regularizer
   function gaussians:init(x, p)
      -- Run K means algorithm to initialize the gaussians
      local km = kmeans(n,k);
      km:init();
      km:learn(x);

      -- Compute the responsibilities using K means
      local r = torch.zeros(x:size()[1], k)	
      for m = 1, x:size()[1] do
	 --Get the owner of this sample
	 o = km:g(x[m])
	 --Set the responsibility
	 r[m][o] = 1
      end	
      --Create and initialize the means and covariances of k Gaussians 
      for j = 1, k do
	 -- Create a Gaussian Object
	 gaussians[j] = gaussian(n);
	 -- Set the mean and covariances for this Gaussian
	 gaussians[j]:learn(x, r:select(2, j), p);
	 -- Initialize the weights
	 weights[j] = torch.sum(r:select(2,j));	
      end  
      		
      -- Normalizing the weights
      weights = weights / torch.sum(weights)   	

      --Print the initial log likelikhood
      print("Initial Negative log likelihood = "..(-1.0 * gaussians:compute(x)));


   end
   
    --The decision function which returns a vector of k elements indicating each gaussian's likelihood
   function gaussians:g(x)
      gaussian_lhds = torch.zeros(k)
      for i = 1, k do 
	gaussian_lhds[i] = gaussians[i]:eval(x)
      end	
      return gaussian_lhds
   end 
 
   --The output function to output a prototype that could replace vector x.
   function gaussians:f(x)
      --Get the likelihoods
      class_likelihoods = gaussians:g(x)
      --Multiply likelihoods with corresponding class weights
      class_likelihoods:cmul(weights); 		
      --Normalize the responsibilities
      class_likelihoods = class_likelihoods / torch.sum(class_likelihoods);
      --Take a weight mean of all the gaussians based on above class_responsibilities
      weighted_mean = torch.zeros(n);
      for j = 1, k do 
	weighted_mean = weighted_mean + (gaussians[j].m) * (class_likelihoods[j]);
      end 	
      return weighted_mean  	
   end



   --Computes the log likelihood of the data based on current parameters
   function gaussians:compute(x)
      local llh = 0
      --Compute 
      for m = 1, x:size()[1] do
           --For this sample, go through all the classes
           sample_sum = 0;
           for j = 1, k do
              sample_sum = sample_sum + (weights[j] * gaussians[j]:eval(x[m]));
           end
           --Add the log likelihood of this sample
             llh = llh + math.log(sample_sum);
      end
      
      --Return the likelihood
      return llh 	
   end

   --Learn the gaussians using x, which is an m*n matrix representing m data samples. p is regularization to keep each gaussian's
   --covariance matrices non-singular. eps is a stop criterion.
   function gaussians:learn(x,p,eps)   
     -- Responsibilities
     local r = torch.zeros(x:size()[1], k)
     -- Log likelihood of the data 
     local old_llh = 0; 	
     local new_llh = 0;	     
     convergence_counter = 0;

 
     --Apply the EM Algorithm
     while true do
	--Set the likelihood to zero
	new_llh = 0;

	--E-Step : Calculate the Responsibilities
  	for m = 1, x:size()[1] do 
	   for j = 1, k do 		
	      r[m][j] = weights[j] * gaussians[j]:eval(x[m]);
	   end

	   if torch.sum(r[m]) ~= 0 then
	     --Normalize the responsibilities for this sample
	     r[m] = r[m] / torch.sum(r[m]);
	   end
	end		


	--M-Step : Update the mean & covariance of the Gaussians and weights
        for j = 1, k do
          -- Learn the mean and covariances for this Gaussian
          gaussians[j]:learn(x, r:select(2, j), p);
          -- Update the weights
          weights[j] = torch.sum(r:select(2,j));
        end
        -- Normalizing the weights
        weights = weights / torch.sum(weights)

		
	--Compute the log likelihood of the data 
	new_llh = gaussians:compute(x)	
	
        --Checking for convergence of log likelihood			
	--print("Difference = "..(math.abs(new_llh - old_llh) / old_llh) * 100)

	if ((math.abs(new_llh - old_llh) / old_llh) * 100) < eps then
	   convergence_counter = convergence_counter + 1	     
	   print "Converged"
	else
	   convergence_counter = 0
	end
			
	--If three successive converges break
	if convergence_counter == 3 then
	   break
	end	

	--Save the old likelihood
	old_llh = new_llh
	print("Negative Likelihood = "..(-1.0 * old_llh))
     end 		
	
  
   end

   --Return this object
   return gaussians

end
