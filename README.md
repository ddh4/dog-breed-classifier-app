# Dog Breed Classifier Application

### Table of Contents
1. [Installation](https://github.com/ddh4/dog-breed-classifier-app#installation)
2. [About](https://github.com/ddh4/dog-breed-classifier-app#about)
3. [Project Motivation and Definition](https://github.com/ddh4/dog-breed-classifier-app#project-motivation-and-definition)
4. [File Descriptions](https://github.com/ddh4/dog-breed-classifier-app#file-descriptions)
5. [Project Summary and Results](https://github.com/ddh4/dog-breed-classifier-app#project-summary-and-results)
6. [Refinement](https://github.com/ddh4/dog-breed-classifier-app#refinement)
7. [Potential Improvements](https://github.com/ddh4/dog-breed-classifier-app#potential-improvements)
8. [Reflection](https://github.com/ddh4/dog-breed-classifier-app#reflection)
9. [Instructions](https://github.com/ddh4/dog-breed-classifier-app#instructions)
10. [Licensing, Authors, and Acknowledgements](https://github.com/ddh4/dog-breed-classifier-app#licensing-authors-and-acknowledgements)

### Installation
The libraries necessary are:
- Flask==1.0.2
- Werkzeug==0.14.1
- opencv_python==4.0.0.21
- Keras==2.0.8
- numpy==1.15.4
- tqdm==4.28.1
- scikit_learn==0.20.2

All libraries necessary are bundled with Anaconda distribution of Python 3.
They can also be installed using pip and requirements.txt.

### About
The final output of this project is a web app where a user can upload a picture of a dog or human which will then be used as input to a CNN classifier. The classifier will return the most likely dog breed resembled in the image. The project contains images for 133 target dogs, one of which will be shown to the user on classification of their image.

The below image is the home page of the application which contains instructions and example output for each of the three cases: 
1. A dog was detected
2. A human was detected 
3. No dogs or humans were detected.

![app_home](https://user-images.githubusercontent.com/39163121/52179638-c9944c00-27d4-11e9-9643-c091df297562.png)

The image below is the prediction page of the application which outputs the classification result for the uploaded image. 

![app_predict](https://user-images.githubusercontent.com/39163121/52179645-d9ac2b80-27d4-11e9-9747-283ff5229e42.png)

### Project Motivation and Definition

The motivation for this project was inspired by a previous Udacity nanodegree project in which we built a classifier to identify types of flower. 

In this project we built a CNN (convolutional neural network) which is a class of deep neural networks, most commonly applied to analyzing visual imagery, to identify dog breeds given an image as user input. 

We have built a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine's breed. If supplied an image of a human, the algorithm will identify the resembling dog breed.

Our pipeline consists of:
- Extracting training data from a series of directories and subdirectories for each dog breed, resulting in a training dataset of 6680 images.
- Transforming the images into a format interpretable by Keras. Keras CNNs require a 4D array/tensor as input, with shape ```(nb_samples,rows,columns,channels)``` where nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively.
- We therefore resize the images to 224x224 pixels and convert the image to a 4D tensor with 3 channels (RGB).
- We then reorder the channels from an RGB image to a BGR image.
- The tensors are then normalized by subtracting the mean pixel from each pixel in the images to center the data and ensure they're on the same scale.
- We then define our model architecture and train using categorial cross entropy as our loss function. 

We do not perform other data augmentation such as rotation, flipping or cropping which is discussed in the [potential improvements section](https://github.com/ddh4/dog-breed-classifier-app#potential-improvements) of the README.

Additionally, a flask application was built which provides an straight forward user experience to utilise the predictive power of the developed CNN. 


### File Descriptions

The jupyter_notebook folder contains the jupyter notebooks used to explore and train our CNN model, as well as functions to detect dogs and faces. Datasets for training, test and validating the CNN models in the notebook are available [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

The app folder contains the files neccessary to run the Flask application, loading using run.py which renders the html templates.

The app/predict folder contains the files neccessary to perform predictions, such as loading the models from memory, detecting dogs, humans and outputing breeds.

The app/static folder contains css and static images which are rendered by the flask application.

The app/templates folder contains the html templates which are rendered by the flask application.

The app/upload_folder is the directory which contains images uploaded to the application by a user. 

### Project Summary and Results 

The CNN developed as part of this project was created using transfer learning, bottleneck features generated from a pretrained Resnet50
on the ImageNet database. Using transfer learning we can reuse complex features that have already been generate so that we do not need to train a model from scratch. These bottleneck features are then input into a global average pooling layer and subsequently into a fully connected dense layer with a softmax activation function to determine which class to predict. 

1. ``GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:])``
2. ``Dense(133, activation='softmax')``

Our model was trained for 20 epochs on 6680 training dog images, validated on 835 dog images and tested on 836 dog images and resulted in an accuracy of 81%. 

Our primary metric during training is categorical cross entropy as firstly, this is a multiclass classification problem. Categorical cross entropy is a good metric as we are using a softmax activation function in the last layer of our CNN and that we are returning a probability distribution. 

The training dataset is slightly imbalanced as the number of training images for each target class in our dataset ranges from 26 to 77, with a mean and median of 50. Therefore some classes are comparatively under-represented, but not severely.

The testing dataset is also slightly imbalanced with the number of images for each target class in our dataset ranging from 3 to 10, with a mean and median of 6. This doesn't negatively impact performance, but it does grant us confidence knowing we have testing data from all classes and therefore will not suffer from the accuracy paradox.

Given the distributions of the dataset, accuracy is an appropriate metric for evaluating this problem as there are 133 target classes and the probability of getting a correct prediction by random guessing is 1/133 (0.07%) and extremely unlikely.

### Refinement 
We started to build a CNN from scratch, defining our model architecture as a series of convolutional and pooling layers, ending with a global averaging pooling layer and a fully connected dense architecture with a softmax activation function. Image classification is a hard task and requires many layers and abundant processing capabilities utilising an individual or multiple GPUs. The peformance of this model left much to be desired and we therefore opted for a transfer learning approach. 

The benefits of transfer learning is that we can utilise the power of pretrained models for certain tasks. For instance, we utilised a prebuilt Resnet model with 50 layers [(Resnet50)](https://arxiv.org/abs/1512.03385) - a deep residual learning model for image recognition. We combined the pretrained weights from Resnet50 with a keras sequential model whereby we defined a global average pooling layer with an input shape matching the output shape of the Resnet50 model. We then defined a a fully connected dense architecture with a softmax activation function for the 133 target classes in our dataset.

I experimented with adding additional layers to the architecture, epochs and optimizer. I kept the best model and parameters in the Jupyter notebook in the repository. Additional improvements for the model are detailed in the next section.

### Potential Improvements
1. Firstly, we could use more images in the datasets used for training, testing and validation. 
2. If more data isn't available, we could further augment the currently available training, testing and validation sets. This can be achieved by applying various transformers, such as rotating, flipping and cropping. This ensures our model is robust and more likely to generalise. 
3. We could add more epochs to further train the model and increase validation accuracy at the expense of longer training times.
4. We could add dropout layers to prevent overfitting.
5. We could add more convolution and pooling layers to extract more features.
6. We could make an ensemble of models and not just rely on the prediction of one. The classification could then be resolved through majority voting.

### Reflection

#### Summary
To summarize, we built a CNN using transfer learning from a Resnet50 model in combination with a keras sequential model. Our algorithm firstly detects whether there exists a dog in the user supplied image, and if not, then detects whether there exists a human. The dog detector uses a pretrained Resnet50 model and the human detector uses a Cascade Classifier in OpenCV.
- If a dog is detected, we make a prediction using our CNN classifier and return the most likely breed.
- If a human is deteced, we make a predictrion using our CNN classifier and return the breed the human mostly resembles.
- If no dog or human is detected, we return an error message.

#### Dicussion/Learning Points
A key learning point was understanding the difficulty and nuances in machine learning tasks such as image classification and how we can deal with these complexities. A related example, dog breed classification is challenging due the similarity of some breeds, such as a Brittany and Welsh Springer Spaniel. CNNs are exceptionally good for these tasks as they can begin to identify complex patterns in our training data as we get deeper into the model architecure. This is also why our model from scratch did not perform so well, due to the limited layers only identifying simple features such as lines and corners.

Another key learning point was how we can utilise the work already achieved through the power of transfer learning. Transfer leaning utilises models that have already been trained on extremely large datasets, providing pretrained weights that can be used for applications such as the one in this project. Transfer learning is extremely beneficial if you don't have a lot of training data available, or if there exists a network for a similar task.


### Instructions
1. Run the following command in the app's directory to run the web app. 
    ``python run.py``

2. Go to http://127.0.0.1:5000/ in your browser.

### Licensing, Authors, and Acknowledgements
Thanks to Udacity for the proposed project and accompanying CNN notebook. The code in this repository can be used freely.
