from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import load_model
from tqdm import tqdm
from predict.extract_bottleneck_features import *
import pickle
import cv2
from keras import backend

import keras; print(keras.__version__)


def Resnet50_predict_breeds(img_path, model):

    with open('predict/models/dog_names.pickle', 'rb') as handle:
        dog_names = pickle.load(handle)

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    #
    #bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)

    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path, model):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(model.predict(img))


def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('predict/models/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    prediction = ResNet50_predict_labels(img_path, ResNet50_model)
    return ((prediction <= 268) & (prediction >= 151))


def dog_breed_predictor(image_path):

    backend.clear_session()

    model_checkpoint = 'predict/models/weights.best.Resnet50.hdf5'
    model = load_model(model_checkpoint)

    dog_detected = dog_detector(image_path)
    human_detected = face_detector(image_path)

    if dog_detected:
        prediction = Resnet50_predict_breeds(image_path, model).split('.')[1].replace("_", " ")
        return "Hello, Dog!", "Your predicted breed is...", prediction
    elif human_detected:
        prediction = Resnet50_predict_breeds(image_path, model).split('.')[1].replace("_", " ")
        return "Hello, Human!", "Your predicted breed is...", prediction
    elif not dog_detected and not human_detected:
        prediction = None
        return "Oops!", " I could not find a dog or a human in the image.\nPlease provide an image of a dog or of a human with a visible face.", prediction
