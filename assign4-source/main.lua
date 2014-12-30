--[[
Sample Main File
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file shows an example of using the tile utilities.
]]

require("image")
dofile("tile.lua")
dofile("kmeans.lua")
dofile("mog.lua")


--Takes "data" and "k" as the input
function doKmeans(t, k)
   local km = kmeans(t:size()[2], k);
   km:init();
   km:learn(t);
   
   local new_image = torch.Tensor(t:size());
   --Getting a prototype
   for m = 1, t:size()[1] do 
     --Go through this tile
     new_image[m] = km:f(t[m])     		
   end
  
   --Return the new image 
   return new_image
end

--Take new image and the old image and returns the mean square error
function calculateMeanError(old,new) 
   local mse = 0;

   for m = 1, old:size()[1] do
      mse = mse + torch.pow(torch.dist(old[m],new[m]),2);
   end

   mse = mse / (old:size()[1])
   return mse
end


function doGMM(t,k)
  local mg = mog(t:size()[2],k);
  local p = 1e-4;
  local eps = 1e-3;
  mg:init(t, p);
  mg:learn(t, p, eps);

  local new_image = torch.Tensor(t:size());
  --Getting a prototype
  for m = 1, t:size()[1] do
   --Go through this tile
   new_image[m] = mg:f(t[m])
  end

  --Return the new image 
  return new_image

end

function testKmeans(t) 
  --Check K means
  k = {2,4,8,64,256};
  for i = 1, table.getn(k) do
    new_image = doKmeans(t,k[i])
    -- Convert back to 800*600 image with 8x8 patches
    im2 = tile.tileim(new_image,{8,8},{600,800})
    -- Show the image
    image.display(im2);
    tile.imwrite(im2, "bird-k"..k[i]..".jpg");
    print(" K = " ..k[i].." MSE = "..calculateMeanError(t, new_image));
  end
end

function Kmeans_Q3(t)
   local km = kmeans(t:size()[2], 8);
   km:init();
   km:learn(t);

   --Histogram Vector .H_k[i] returns the number of elements whose closest prototype had a label 'i'
   local H_k = torch.zeros(8);
   --Getting a prototype
   for m = 1, t:size()[1] do
     --Go through this tile
     index = km:g(t[m]);
     H_k[index] = H_k[index] + 1;
   end
   --Normalize 
   H_k = H_k / (torch.sum(H_k))
   
   --Calculate the Histogram Entropy
   HE = 0;
   for k = 1, 8 do
      HE = HE + (H_k[k] * (math.log(H_k[k]) / math.log(2)) * -1.0);
   end

   print("HE = " ..HE);
   --Total no of bits required  
   print("Total no of bits required = " ..(t:size()[1] * HE))   

end



function testGMM(t)
  --Check K means
  --k = {2,4,8,64,256};
  k = {8};
  for i = 1, table.getn(k) do
    new_image = doGMM(t,k[i])
    -- Convert back to 800*600 image with 8x8 patches
    im2 = tile.tileim(new_image,{8,8},{600,800})
    -- Show the image
    image.display(im2);
    tile.imwrite(im2, "bird-gmm"..k[i]..".jpg");
    print(" K = " ..k[i].." MSE = "..calculateMeanError(t, new_image));
  end
end



-- An example of using tile
function main()
   -- Read file
   im = tile.imread('boat.png')
   -- Convert to 7500*64 tiles representing 8x8 patches
   t = tile.imtile(im,{8,8})
   
   testKmeans(t);
   --testGMM(t)
   --Kmeans_Q3(t)

end

main()
