import json
from flask import Flask
from flask import render_template, request, jsonify, send_from_directory
import numpy as np
from werkzeug.utils import secure_filename
import os
from glob import glob
from predict.predict import dog_breed_predictor

app = Flask(__name__)
UPLOAD_FOLDER = 'upload_folder'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/predict', methods=['POST'])
def predict():

    print(request.files)

    uploaded_file = request.files['uploaded_file']

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        greeting, message, prediction = dog_breed_predictor(file_path)
        # PREDICTION FUNCTIONALITY
        # Remove image from UPLOAD_FOLDER
        if prediction is not None:
            dog_names = [item.split('\\')[-1] for item in sorted(glob("static\\images\\dog_breeds\\*"))]
            breed_file = [d for d in dog_names if prediction.replace(" ", "_") in d]
        else:
            prediction = "No Classification"
            breed_file = ['no_classification.jpg']


    # This will render the go.html Please see that file.
    return render_template(
        'predict.html',
        prediction_greeting = greeting,
        prediction_message = message,
        prediction_result = prediction,
        uploaded_image = filename,
        breed_image = "images/dog_breeds/{}".format(breed_file[0])
    )

# https://stackoverflow.com/questions/11262518/how-to-pass-uploaded-image-to-template-html-in-flask
@app.route('/uploads_folder/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)





def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
