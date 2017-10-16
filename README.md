#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

####1. Provide README that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/malnakli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and defualt python  to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided add additinal data because the variance between the dataset is very high which is 392816, and the mean is 809. Therefore, I decided to use agumtation in order to generate fake data. the [imgaug](http://imgaug.readthedocs.io/en/latest/index.html ) library was chosen to facilitate the agumtation. 
There many techniques were select look at **augment_data(images)** for more info. Here is some of them
rotatation: to teach the network how to recover from a poor position. 
CropAndPad: I put a small percetage (-0.25, 0.25) in order to make sure the sign still present on an image, but a small piece of the sign is missing. 
ChangeColorspace: I found that changing the color space it give better result.

Here is an example of an original image and an augmented image:



![original augmented][]

As a second step, I normalized the image data because If a feature has a variance that is orders of magnitude larger than others, it may  make the estimator unable to learn from other features correctly as expected.


I decided to convert the image to gray because I want to teach the netwrok that color does not  matter on determining the trafic sign

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an gray image:

![original gray][]

The difference between the original data set and the augmented data set is the following         
Number of old training examples is 34799         
Number of new training examples is  135342         
Image data shape is  (135342, 32, 32, 3)       
old mean = 809      
new mean = 3147       
old variance = 392816       
new variance = 92519      
#########################  number of each sign occurrence #########################    
| ClassID         		|     Occurrence	|  Name |
|:--------:|:------------------:|:--------------------------------------:|  
| 5.0	 | occurrence =3964	| Speed limit (80km/h)  |   
| 25.0	| occurrence =3806	|Road work|
|1.0	| occurrence =3683|	Speed limit (30km/h)|
|13.0	| occurrence =3624|	Yield|
|12.0	| occurrence =3551|	Priority road|
|35.0	| occurrence =3549|	Ahead only|
|2.0	| occurrence =3478|	Speed limit (50km/h)|
|9.0	| occurrence =3386|	No passing|
|14.0	| occurrence =3298|	Stop|
|38.0|	 occurrence =3277	|Keep right|
|17.0|	 occurrence =3266|	No entry|
|16.0|	 occurrence =3210|	Vehicles over 3.5 metric tons prohibited|
|7.0|	 occurrence =3202|	Speed limit (100km/h)|
|11.0|	 occurrence =3196|	Right-of-way at the next intersection|
|28.0|	 occurrence =3174|	Children crossing|
|8.0|	 occurrence =3164|	Speed limit (120km/h)|
|31.0|	 occurrence =3162|	Wild animals crossing|
|21.0|	 occurrence =3138|	Double curve|
|4.0|	 occurrence =3132|	Speed limit (70km/h)|
|10.0|	 occurrence =3123|	No passing for vehicles over 3.5 metric tons
|20.0|	 occurrence =3120|	Dangerous curve to the right|
|34.0|	 occurrence =3110|	Turn left ahead|
|39.0|	 occurrence =3104|	Keep left|
|22.0|	 occurrence =3090|	Bumpy road|
|26.0|	 occurrence =3090|	Traffic signals|
|19.0|	 occurrence =3087|	Dangerous curve to the left|
|27.0|	 occurrence =3075|	Pedestrians|
|3.0|	 occurrence =3054|	Speed limit (60km/h)|
|29.0|	 occurrence =3035|	Bicycles crossing|
|24.0|	 occurrence =3026|	Road narrows on the right|
|40.0|	 occurrence =3020|	Roundabout mandatory|
|33.0|	 occurrence =3014|	Turn right ahead|
|18.0|	 occurrence =3010|	General caution|
6.0	 |occurrence =3000|	End of speed limit (80km/h)|
36.0|	 occurrence =2992|	Go straight or right|
30.0|	 occurrence =2990|	Beware of ice/snow|
15.0|	 occurrence =2976|	No vehicles|
23.0|	 occurrence =2970|	Slippery road|
41.0|	 occurrence =2964|	End of no passing|
37.0|	 occurrence =2679|	Go straight or left|
0.0	 |occurrence =2594	|Speed limit (20km/h)|
42.0|	 occurrence =2590|	End of no passing by vehicles over 3.5 metric tons|
32.0|	 occurrence =2369|	End of all speed and passing limits|



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   	| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 |
| RELU					|	 Activation	 function					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6|
| Convolution 3x3	    |  1x1 stride, VALID padding, outputs 10x10x16|
| RELU					|		Activation	 function	       		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                 |
| Flatten               | outputs 400                               |
| Fully connected		| outputs 120        						|
| RELU					|		Activation	 function	       		|
| Fully connected		| outputs 80        						|
| RELU					|		Activation	 function	       		|
| dropout               | 50% in order to smath the curve |
| Softmax				| output = 43      								
|                       |	                        |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




https://arxiv.org/pdf/1604.04004.pdf
https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/