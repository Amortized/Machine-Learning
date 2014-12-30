Linear Models Assignment Description
Written by Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com)
Version 0.1, 09/24/2012

This file is written for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Your assignment should be sent by email to the TA
- Send all the source code files (.lua) plus a description in one zip or
tar.gz. Please excluding datasets (.t7, .data), or your email may not be
received.
- The description can be a simple text file, with any mathematical notion
(if there is any) written in LaTeX convention. But please do NOT include any
msword or html files. If you do not like writing formulas in LaTeX convention,
we can accept pdf if it is small. Write all your answers to the QUESTIONS
section (below) in this description file.
- You must implement the code by yourself, but you may discuss your results
with other students.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following learning algorithms must be implemented with ridge (l2) and
LASSO (l1) regularization:
- The perceptron algorithm
- Direct solution of linear regression
- On-line (stochastic) linear regression
- On-line (stochastic) logistic regression
- On-line (stochastic) multinomial logistic regression

In order to implement the programs, you must download the datasets for this
assignment (available form the course website), and put all files in the same
path as your source code. Notice that for the mnist dataset we are using a
different file (native to torch) than that provided on its website.

The dataset with which to test the binary classification models is called
uci-spambase. It is a two-class classificiation problem from UC Irvine
machine learning dataset repository
(http://www.ics.uci.edu/~mlearn/MLRepository.html):
The dataset contains 4601 samples with 57 variables. The task is to predict
whether an email message is spam or not.

The datset with which to test the multinomial model is called MNIST
(http://yann.lecun.com/exdb/mnist/). It has a training set of 60,000 examples,
and a test set of 10,000 examples. The task is to classify each 32x32 image
into 10 categories (corresponding to digits 0-9).
							
The torch code to read the datasets and format them is provided. Please refer
to spambase.lua and mnist.lua. Upon calling the getDatasets() methods, all the
data will be appended with an extra dimension which will constantly be 1, so
that you do not need to worry about the bias when implementing a linear model.

You simply need to edit the files main.lua, model.lua, regularizer.lua,
trainer.lua and implement the algorithms. The detailed design paradigm is
illustrated in the heading comments of each file. Please read them. Examples
are given for each kind of object you will have to implement.

The places where you should implement something is identified by the follwoing
two lines:
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");

Your training code should
- Starting by computing and displaying loss and the error rate on the training
set and test set.
- Make a training pass over the training set
- Compute and display the loss and error rate on the training set and test set.
- Iterate to 2 until convergence (pick a criterion).

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

QUESTIONS:

1 - Implement:
  a. - The perceptron learning algorithm
		
        
  b. - Linear regression with square loss trained with direction solution and 
       stochastic gradient descent


  c. - The Logistic regression algorithm trained with stochastic gradient
       descent


  d. - The multinomial Logistic regression algorithm trained with stochastic
       gradient descent

  ==> Include your code as attachment of an email to the TA.

2 - Experiments with the Spambase dataset
     - Set the training set size to 1000 and the test set to 1000.
     - Use the (ridge) l2 regularization
     - The stochastic gradient method for linear regression and logistic
       regression requires you to find good values for the learning rate
       (the step size eta using stepCons)


	For 100 Iterations
	Linear Reg - 
	0.0003 Training loss = 0.20017870675487, error = 0.097; Testing loss = 0.28846400001774, error = 0.112
        0.0004 Training loss = 0.20435783833232, error = 0.095; Testing loss = 0.31823071068508, error = 0.112	
  	0.0005 Training loss = 0.21806383758719, error = 0.1; Testing loss = 1.8913831778705, error = 0.086
	0.0008 Training loss = 0.20627200000231, error = 0.103; Testing loss = 0.22872381655857, error = 0.109
	0.001  Training loss = 0.2111225832508, error = 0.116; Testing loss = 0.2385982242151, error = 0.116
	0.002  Training loss = 14109.368351353, error = 0.407; Testing loss = 4105.4023718328, error = 0.453
	0.003  Training loss = 9.279898822597e+36, error = 0.516; Testing loss = 2.4831021545637e+36, error = 0.482
        0.005  Training loss = 1.3853187495339e+40, error = 0.485; Testing loss = 2.939511238814e+40, error = 0.494
        0.008  Training loss = 9.8001871023428e+48, error = 0.372; Testing loss = 4.1919719922828e+48, error = 0.376
	


	Logistic Reg - 
	0.001  Training loss = 1.2887082998885, error = 0.118; Testing loss = 1.2804550885335, error = 0.117
	0.002  Training loss = 1.2531188327886, error = 0.157; Testing loss = 1.2590714907106, error = 0.165
	0.003  Training loss = 1.2091567612621, error = 0.126; Testing loss = 1.2167131772108, error = 0.143
	0.004  Training loss = 1.1527606988025, error = 0.098; Testing loss = 1.1610269639319, error = 0.1
	0.005  Training loss = 1.1133749721157, error = 0.142; Testing loss = 1.1258594507268, error = 0.149
	0.006  Training loss = 1.1480359994861, error = 0.151; Testing loss = 1.1371680386093, error = 0.148
	0.008  Training loss = 1.0515408718359, error = 0.156; Testing loss = 1.0882636269119, error = 0.16
	0.009  Training loss = 1.0582014393977, error = 0.141; Testing loss = 1.0617532225705, error = 0.143
	0.01   Training loss = 1.0326725007256, error = 0.152; Testing loss = 1.0396702771979, error = 0.171




 
	- 



  a. - What learning rate will cause linear regression and logistic regression
       to diverge?		


       Divergence happens:
	 Linear Regression : 0.002(Test Error jumps up by 34%)
	 Logistic Regression : 0.005(Test Error jumps up by 5%)
	

  b. - What learnin rate for linear regression and logistic regression that
       produces the fastest convergence?

       Fastest convergence 
       Assuming a Constant Step Size and L2 reg with lambda as 0.05

       0.0005 -linear
       0.004  -logistic     	

  c. - Implement a stopping criterion that detects convergence.

	Convergence was decided on the basis of euclidean distance between the new model and model predicted in the previous 
	stage.Threshold for the distance was choosen as 0.001. If this happens three times consecutively  it is interpreted 
        as convergence

	Results
	Linear -  Training loss = 0.21074346950595, error = 0.107; Testing loss = 0.26177948690963, error = 0.114
	Logistic - Training loss = 1.1233763957545, error = 0.115; Testing loss = 1.111909545327, error = 0.121
 

  d. - Train logistic regression with 10, 30, 100, 500, 1000 and 3000 trianing
       samples, and 1000 test samples. For each size of the training set,
       provide:
       - The final value of the average loss, and the classification errors on
         the training set and the test set
       - The value of the learning rate used
       - The number of iterations performed

Samples     Training Rate           Number of Iterations Used                 Error Values
10          0.004			100			 Training loss = 0.8400, error = 0; Testing loss = 3.1719,error = 0.339
				        125			 Training loss = 0.6220, error = 0; Testing loss = 8.4564, error = 0.324
					150   			 Training loss = 1.1222, error = 0; Testing loss = 4.6989,error = 0.371


30          0.004                       100                      Training loss = 0.7994, error = 0; Testing loss = 1.8346,error = 0.292
					125			 Training loss = 0.8644, error = 0; Testing loss = 1.2388,error = 0.268
 					150		         Training loss = 0.9553, error = 0; Testing loss = 1.3256,error = 0.235
 

100         0.004			100			Training loss = 1.3168, error = 0; Testing loss = 1.3357, error = 0.217
					125			Training loss = 1.3625, error = 0; Testing loss = 1.3600, error = 0.181
					150			Training loss = 1.3637, error = 0; Testing loss = 1.3679, error = 0.17

500 	    0.004			100		      Training loss = 1.1003, error = 0.102; Testing loss = 1.1110, error = 0.139 						125	              Training loss = 1.1703, error = 0.104; Testing loss = 1.1843, error = 0.138
 					150	              Training loss = 1.1591, error = 0.086; Testing loss = 1.1906, error = 0.153


1000        0.004			100		   Training loss = 1.1470, error = 0.121; Testing loss = 1.1390, error = 0.118
					125		   Training loss = 1.1176, error = 0.082; Testing loss = 1.1428, error = 0.111
					150 		   Training loss = 1.1490, error = 0.121; Testing loss = 1.1600, error = 0.129

3000        0.004			100		Training loss = 1.1496, error = 0.137; Testing loss = 1.1597, error= 0.152 						125		Training loss = 1.1494, error = 0.14466666666667; Testing loss = 1.1523, error = 0.155
 					150	        Training loss = 1.1646, error = 0.095333333333333; Testing loss = 1.1659, error = 0.101


  e. - What is the asymptotic value of the training/test error for very large
       training sets?
  
     - A     


3. - L2 and L1 regularization
     When the training set is small, it is often helpful to add a
     regularization term to the loss function. The most popular ones are:
     L2 Norm: lambda*||W||^2 (aka "Ridge")
     L1 Norm: lambda*[\sum_{i} |W_i|] (aka "LASSO")
  a. - How is the linear regression with direct solution modified by the
       addition of an L2 regularizer?



  b. - Implement the L1 regularizer. Experiment with your logistic regression
       code with the L2 and L1 regularizers. Can you improve the performance on
       the test set for training set sizes of 10, 30 and 100? What value of
       lambda gives the best results?

		
------- STOCHASTIC GRADIENT DESCENT  STEPSIZE : 0.0005        Value of Lambda was incremented from 0.001 to 1 with a increase of 0.001------



------  Best Results are the one which have min training error, min training loss, min test error and min test loss ----

	No of training samples    Iterations                L1 Reg         Values          

		10		   100	   	Best Lambda = 0.003   Training loss = 0.8793, error = 0; Testing loss = 5.4926, error = 0.351  
						            = 0.05  	 
						            = 1      	
				

				   500          Best Lambda = 0.003  Training loss = 0.7705, error = 0.1; Testing loss = 12.2032, error = 0.394


											
		30	           100      	Best Lambda = 0.001   Training loss = 0.4248, error = 0; Testing loss = 2.6825, error = 0.211 
						            = 0.05 
						            = 1     

				
				   500


							

		100		   100        Best Lambda = 0.001 Training loss = 0.5362, error = 0.1; Testing loss = 1.9084, error = 0.185						                 = 0.05
						          = 1


		  		
 				   500
				  

	



4. - Multinomial logistic regression
     Implement the multinomial logistic regression as a model. Your can
     experiment with larger datasets if your machine has enough memory. In this
     part we experiment on the MNIST dataset.
  a. - Testing your model with training sets of size 100, 500, 1000, and 6000,
       with a testing set of size 1000. What are the error rates on the
       training and testing datasets?
  b. - Use both L2 and L1 regularization. Do they make some difference?
