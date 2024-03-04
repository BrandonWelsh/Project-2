# Project 2, Group 8
- Arnab Roy
- Brandon Welsh
- Sunny Kiluvia

February/March 2024

Due date: March 4th 2024
## Program Goals
On this project we will practice both our technical and collaborative skills. We will work in groups of 3 people. The project will be divided in 3 parts: Data Exploration, Data Cleaning and Feature Engineering, and Model Selection and Evaluation. We will collaborate with our group members to complete the project. We will also be responsible for our own individual submissions.

We decided to pursue a project on image classification, as this topic was of interest to all of us. We found a dataset (details in next section) that contained images of galaxies, and plan to use this to train a variety of classification algorithms and convolutional neural networks in an attempt to maximize the train and test scores (aiming for at least 75% classification accuracy) and obtain a model which can reliably classify galaxies as being one of four types: 

0 - smooth and round 

1 - smooth and cigar-shaped

2 - edge-on-disk

3 - unbarred spiral

## Data Source
We were very lucky with the data that we found. It contained 10,000 images and already came cleaned, labeled, and split into train and test datasets. This saved us a lot of time and allowed us to dive right in and start training models. Utilizing a file called galaxy_mnist.py, we were able to download these datasets as hdf5 files into a local Resources folder, and were able to define variables for the training data (8,000 images saved as a 64x64 pixel 3D array, pre-labeled 0 thru 3, corresponding to each of the four galaxy types) and the testing data (2,000 images unique from the ones in the training dataset, but otherwise had all the same characteristics).

The data source for our project comes from the following github repo:

- https://github.com/mwalmsley/galaxy_mnist

## Dependencies/Setup Instructions
You will need the following Python libraries, as well as the galaxy_mnist.py file (included in this github repository). You will also need to have each of the libraries required to run the following lines of code contained within the Project-2.ipynb file:
- from galaxy_mnist import GalaxyMNIST
- import matplotlib.pyplot as plt
- import numpy as np
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.metrics import accuracy_score
- from sklearn.svm import SVC
- from sklearn.linear_model import LogisticRegression
- from sklearn import tree
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.ensemble import ExtraTreesClassifier
- from sklearn.ensemble import GradientBoostingClassifier
- from sklearn.ensemble import AdaBoostClassifier
- from tensorflow.keras.models import Sequential
- from tensorflow.keras.layers import Conv2D, Flatten, Dense
- from tensorflow.keras import regularizers
- from tensorflow.keras.layers import Dropout
- from tensorflow.keras.optimizers import Adam
- from tensorflow.keras.utils import to_categorical
- from tensorflow.keras import models, layers, regularizers

## How to use
Run each cell of the Project_2.ipynb file to create and train the models used in this project. They should have already been run when uploaded to github and are ready to be viewed.

## Team Member Responsibilities
Since our data came pre-prepared, this saved us a ton of time and we were able to dive right in and start training models. Taking advantage of this opportunity, we found time to train not one, but eleven unique models. We started with the basic classification algorithms that we covered in class, but soon learned that these models are not well suited for image classification. They required the data to be flattened into a 1D array, which caused it to lose some of its dimensionality (while retaining some patterns which our models were able to capture nonetheless). As a result, we came together and decided to pursue Convolutional Neural Networks, which we have not yet covered in class. This was a considerable challenge, but we were up to the task. We each settled on a simple but unique Convolutional Neural Network (CNN) and tuned the hyperparameters of that model in an attempt to maximize the train and test scores and find the best model for classifying images of galaxies.

Arnab Roy:
- Logistic Regression Classifier
- Support Vector Machine (SVM)
- K-nearest Neighbors (KNN)
- LeNet-5 Convolutional Neural Network

Brandon Welsh:
- Gradient Boosting Classifier
- Adaptive Booster Classifier
- Single-layer Convolutional Neural Network

Sunny Kiluvia:
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Multi-layer Convolutional Neural Network

## Team Member Analysis
TODO: discuss your process for creating all of your models, discuss modifying hyperparameters, and describe your results and their meanings.

### Arnab Roy:
I was tasked with creating and performing a Logistic Regression, Support Vector Machine (SVM), K-nearest Neighbors (KNN), and LeNet-5 Convolutional Neural Network. When we started the execrcise, we kind of knew the first three models, which are conventional machine learning algorithms to perform the classification task, might not be effcient for image classification. We decided to give them a try anyway and then spend some time to try to develop the Neural Network models which can handel galaxy image more efficiently .Irrespective of what model we used, I needed to flatten the raw PIL images to 1D & numpy arrays before I can use them in our models. For LeNet-5 Convolutional Neural Network, it was even needed to normalize the pixel values to be between 0 and 1 before the training data can be used to train the models. Training & Testing data was then used to calculate the Train & Test scores & predict the labels of the testing data.

--- K-nearest Neighbors (KNN) ---
I started with KNN model. I ran a for loop incrementing the number of neighbors by 2, looping through 1 to 15 & calculating the Train & Test scores for each of the neighbors. The train & test accuracy scores were then plotted against the number of neighbors to find the most optimal number of neighbors which was decided to be 9, beyond which the accuracy scores started to stabilize. The Train & Test scores were 0.723 &  0.651 respectively for 9 neighbors. Low Training Accuracy & Test data Accuracy suggests that the model is underfitting the data.


--- Support Vector Machine (SVM) ---
I then moved on to SVM model. Default hyperparameters of Radial Basis Function (RBF) Kernel & C=1 were used for the model. I was able train the model for all 8000 images in the training dataset. With these hyperparameters, we are able to achieve a better Train & Test scores of 0.839 & 0.748 respectively. Even though the Training Accuracy & Test data Accuracy score are still undefitting the data, the model is performing better than KNN model.

--- Logistic Regression Classifier ---
With the help of some reasearch, I kind of stipulated that convergangce of the model might be an issue with the Logistic Regression model. Tried changing the hypermater (number of iterations) from default value of 100 to 
10,000, but the model still didn't convergence after runnin for 10 mins. This might be due to the large volume of feature galaxy image data which we are dealing with. The next logical step to get a solution here was to reduce the number of features which we reduced it to 1000 keeping the max_iter to the same value of 10000. I did achieve the model convergangce finally but its a no brainer that the model is underfitting with such less data Training & Testing Data Scores were 0.54725 & 0.482 respectively.

--- LeNet-5 Convolutional Neural Network ---
It was required to reshape x-train & x-test to the original image dimensions of 64x64x3. And then it was required to normalize the pixel values to be between 0 and 1 before the training data can be used to train the models. 
An optimal value of 3 hyperparameters (epochs=10, validation_split=0.2, batch_size=32) were found for training the model after trying the below combinations of hyperparameters, the results of which hinted overfitting. The most effcient range of validation_split is in the range: 0.1 - 0.3, which didn't alter the results much. We decided to go with the default value of 0.2. 

epochs = 10, batch_size	= 64	
- Training Score: 0.9165
- Testing Score: 0.801
							
epochs = 20, batch_size = 32	
- Training Score: 0.94775
- Testing Score: 0.7895
				
epochs = 5, batch_size = 32	
- Training Score: 0.95525	
- Testing Score: 0.802

Tried following a tedious & dirty was of changing the other hyperparameters like activation function, optimizer, loss function etc. to see if the model can be improved. The best results which we have got is:
manually and the best results which we have got is:
- Training Score: 0.849
- Testing Score: 0.787

------Automating the hyperparameter tuning process using using Keras Tuner:
Keras Tuner offers several tuners to automate 'hyperparameter tuning' in this context of improving our LeNet-5 model. This includes RandomSearch, Hyperband, BayesianOptimization and Sklearn. For this example, we'll use the Hyperband tuner, which is efficient and effective for deep learning models, that involves the below steps as practiced in the code:
Step 1: Define Model-Building Function with Hyperparameters
Step 2: Instantiate the Tuner and Perform Hyperparameter Search
Step 3: Review and Retrieve the Best Model, find the best hyperparameters, train the model and get the scores.

The best results which we have got using this automated process is:
- Training Score: 0.9115
- Testing Score: 0.8155

### Brandon Welsh:
I was tasked with creating and performing a Gradient Boosting and Adaptive Boosting classifier algorithm, as well as a Single-Layer Convolutional Neural network. While the former didn't have hyperparameters I could change, I was able to see a difference in scores by limiting the number of images from the dataset the models were allowed to be trained on. As was expected, the scores increased as the models were given access to more training data. Unfortunately, these models are not well suited for image classification, and as such, their scores were not very high. As a reminder, we have a goal of obtaining 75% or greater test classification accuracy.

--- Gradient Boosting ---

Unfortunately, I quickly found that this model would have taken hours to run if given the full dataset. I first opted to limit it to the first 100 images (1/100th the size of the full dataset), and even this took a few minutes to run. The train and test scores are close to each other, but are relatively low. The model showed improvement when given 200 images rather than just 100, so it can be presumed that the score would be pretty good if given the full set of 10,000. Unfortunately, as previously stated, it would take hours to train that. I bit the bullet and tried 1,000 images. It took half an hour but it saw great improvement. It can be presumed that the score would be pretty good if given the full dataset.

First 100 images:
- Training Score: 0.4065
- Testing Score: 0.41

First 200 images:
- Training Score: 0.5015
- Testing Score: 0.483

First 1000 images:
- Training Score: 0.6995
- Testing Score: 0.6575

--- Adaptive Boosting ---

This model, despite being given the same data, performed worse than the Gradient Boosting. It's important to note that neither of these are well suited for image classification, especially given that the data had to be transformed into a 1D array before these could even be utilized. There was no noticeable improvement between 100 and 200 images. This model ran a little faster so I increased it to 1000 images. Strangely, this is where I saw a considerable improvement, and nearly the same score as Gradient Boosting. I decided to push it to the full dataset, as this model doesn't seem to have the same runtime issue that Gradient Boosting does. Despite having the entire dataset to be trained on, this model didn't perform quite as well as I hoped, with only a 57% testing score on the full dataset.

First 100 images:
- Training Score: 0.323875
- Testing Score: 0.323

First 200 images:
- Training Score: 0.34
- Testing Score: 0.3195

First 1000 images:
- Training Score: 0.508
- Testing Score: 0.4825

Entire dataset:
- Training Score: 0.606125
- Testing Score: 0.5705

--- Single-Layer Convolutional Neural Network ---

I've never built a Convolutional Neural Network (CNN) before, and we haven't even covered it in class yet. Nevertheless, these are the optimal models to use for image classification tasks. I utilized Microsoft Copilot and had it give me a framework for a single-layer CNN using tensorflow. The base model of my CNN performed rather poorly, so I proceeded to alter hyperparameters to seek out the optimal score. I used Microsoft Copilot to guide me through this process, as I wasn't sure which parameters to change. It was greatly helpful as it was able to see my code and gave suggestions which actually helped to improve the score. Some of the hyperparameters I changed included adding L2 Regularization, adding a dropout rate of 0.5 (later changed to 0.3), trying different Activiation methods (ELU was optimal), and then changing the optimizer's learning rate. In the end, I managed to get a training score of 1.0 and a testing score of 0.70. This is a clear sign of overfitting in my model, as the CNN performed great on the training dataset but relatively poorly on the test data. I recorded my progress, changes, and model scores to the following text file. The main project file contains the most optimized CNN that I could come up with during this process. The first and final iteration's scores are displayed below:

[Single Layer CNN.txt](https://github.com/BrandonWelsh/Project-2/files/14475687/Single.Layer.CNN.txt)

Single Layer CNN Iteration 1:
- Training Score: 0.9762499928474426
- Testing Score: 0.5040000081062317

Single Layer CNN Final Iteration:
- Training Score: 1.0
- Testing Score: 0.7009999752044678

While I was able to greatly improve the score of my model, it fell short of the goal of 75% classification accuracy. I made several other hyperparameter modifications following the achievement of this score, but none were able to achieve higher than 0.70 testing score. In addition, my model is clearly overfitting, as the training score is perfect but the testing score is iffy. To conclude, I cannot recommend a Single-Layer CNN be used in this case. It is far too simple of a model to be able to reliably capture the patterns in complex data. Luckily, this is not the only CNN our group created, and it is also expected to perform the worst as it is not a very complex model. On the plus side, this was great practice ahead of actually getting into neural networks in class in the coming weeks.

### Sunny Kiluvia:
TODO

## Resources Utilized

This section is dedicated to keep track of what we used to help complete this project:

Python: The main programming language for our project, including the required libraries as outlined in the Dependencies section above.

Jupyter Notebooks: Used to help keep our project clean and allows the numerous models we wrote to be run independently rather than simultaneously. 

Generative Artificial Intelligence: This is an extrememly powerful tool for programmers and developers. We utilized AI (notably Microsoft Copilot) throughout this project. It assisted with preparing the data for use in the classification algorithms (as we did not realize we needed to flatten the 3D arrays into 1D arrays for these). AI also helped us easily build the convolutional neural networks which we had not yet covered in class, gave us suggestions for hyperparameter tuning, and helped us interpret our results. Throughout the project, it was also utilized to help troubleshoot code as well as how to load in libraries which we have not yet used in class.

## Bugs
This is less of a bug and more of a special note: We opted to each publish our work to our own individual branches and then Brandon (the owner of the repository) would manually add these changes to his branch and merge his branch with main. We opted to do this as a direct result of conflict issues which were becoming progress-halting.

In the process of doing so, Arnab wrote an awesome Keras optimization algorithm for his LeNet-5 Convolutional Neural Network. However, it took 3 hours to run and created several large folders of data, which could not easily be uploaded to github. To circumvent this and prevent Arnab from losing his hard work, we opted to upload his entire Jupyter Notebook as a separate file to this repository. The outputs (including the all-important test and train scores) were preserved, but running the notebook is ill-advised because of how long it would take to run.

## Update Log
Feb 22: Created github repository, shared google drive, and slack channel for the purpose of collaboration and planning. Found dataset.

Feb 26: Initial data exploration and data transformation in preparation of creating classification algorithms. Assigned classification algorithms to each group member.

Feb 28: Finished and double checked Classification Algorithms. Assigned Convolutional Neural Networks to each group member.

Feb 29: Made significant progress on CNNs

Mar 2: Finished Single-layer CNN and LeNet-5 CNN

Mar 3: Finished the main jupyter notebook, README.md, team member write-ups, and google slideshow. Made plans in preparation for presentation on March 4th.

