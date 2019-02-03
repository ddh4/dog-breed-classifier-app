# Dog Breed Classifier Application

### Table of Contents
1. [Installation](https://github.com/ddh4/dog-breed-classifier-app#installation)
2. [About](https://github.com/ddh4/dog-breed-classifier-app#about)
3. [Project Motivation and Definition](https://github.com/ddh4/dog-breed-classifier-app#project-motivation-and-definition)
4. [File Descriptions](https://github.com/ddh4/dog-breed-classifier-app#file-descriptions)
5. [Project Summary and Results](https://github.com/ddh4/dog-breed-classifier-app#project-summary-and-results)
4. [Instructions](https://github.com/ddh4/dog-breed-classifier-app#instructions)
5. [Licensing, Authors, and Acknowledgements](https://github.com/ddh4/dog-breed-classifier-app#licensing-authors-and-acknowledgements)

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

Additionally, a flask application was built which provides an straight forward user experience to utilise the predictive power of the developed CNN. 


### File Descriptions

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

Accuracy is an appropriate metric for this problem as there are 133 target classes and the probability of getting a correct prediction by random guessing is 1/133 (0.07%) and extremely unlikely.

#### Potential Improvements
1. Firstly, we could use more images in the datasets used for training, testing and validation. 
2. If more data isn't available, we could further augment the currently available training, testing and validation sets. This can be achieved by applying various transformers, such as rotating, flipping and cropping. This ensures our model is robust and more likely to generalise. 
3. We could add more epochs to further train the model and increase validation accuracy at the expense of longer training times.
4. We could add dropout layers to prevent overfitting.
5. We could add more convolution and pooling layers to extract more features.
6. We could make an ensemble of models and not just rely on the prediction of one. The classification could be resolved then through majority voting.

### Instructions
1. Run the following command in the app's directory to run the web app. 
    ``python run.py``

2. Go to http://127.0.0.1:5000/ in your browser.

### Licensing, Authors, and Acknowledgements
Thanks to Udacity for the proposed project and accompanying CNN notebook. The code in this repository can be used freely.
