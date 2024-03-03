# Project 2, Group 8
- Arnab Roy
- Brandon Welsh
- Sunny Kiluvia

February/March 2024
## Program Goals
On this project we will practice both our technical and collaborative skills. We will work in groups of 3 people. The project will be divided in 3 parts: Data Exploration, Data Cleaning and Feature Engineering, and Model Selection and Evaluation. We will collaborate with our group members to complete the project. We will also be responsible for our own individual submissions.

We decided to pursue a project on image classification, as this topic was of interest to all of us. We found a dataset (details in next section) that contained images of galaxies, and plan to use this to train a variety of classification algorithms and convolutional neural networks in an attempt to maximize the train and test scores and obtain a model which can reliably classify galaxies as being one of four types: 

0 - smooth and round 

1 - smooth and cigar-shaped

2 - edge-on-disk

3 - unbarred spiral

## Data Source
We were very lucky with the data that we found. It contained 10,000 images and already came cleaned, labeled, and split into train and test datasets. This saved us a lot of time and allowed us to dive right in and start training models. Utilizing a file called galaxy_mnist.py, we were able to download these datasets as hdf5 files into a local Resources folder, and were able to define variables for the training data (8,000 images saved as a 64x64 pixel 3D array, pre-labeled 0 thru 3, corresponding to each of the four galaxy types) and the testing data (2,000 images unique from the ones in the training dataset, but otherwise had all the same characteristics).

The data source for our project comes from the following github repo:

- https://github.com/mwalmsley/galaxy_mnist

## Dependencies
TODO: figure out all dependencies once the main code is complete

## Setup Instructions
TODO: ditto dependencies

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

Arnab Roy:
TODO

Brandon Welsh:
TODO

Sunny Kiluvia:
TODO

## Resources Utilized

This section is dedicated to keep track of what we used to help complete this project:

Python: The main programming language for our project, including the required libraries as outlined in the Dependencies section above.

Jupyter Notebooks: Used to help keep our project clean and allows the numerous models we wrote to be run independently rather than simultaneously. 

Generative Artificial Intelligence: This is an extrememly powerful tool for programmers and developers. We utilized AI throughout this project. It assisted with preparing the data for use in the classification algorithms (as we did not realize we needed to flatten the 3D arrays into 1D arrays for these). AI also helped us easily build the convolutional neural networks which we had not yet covered in class, gave us suggestions for hyperparameter tuning, and helped us interpret our results. Throughout the project, it was also utilized to help troubleshoot code as well as how to load in libraries which we have not yet used in class.

## Bugs
TODO

## Update Log
haha oops

