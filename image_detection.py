from __future__ import print_function  # In python 2.7
# import os
from flask import Flask, request
import sys
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
from pandas import DataFrame, read_csv
from flask import Flask, render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import glob
import shutil
import os
from tensorflow.keras.models import load_model
from src.features.build_features import make_annotation_dataframe
from src.visualization.visualize import eda_dataset
from src.models.train_model import model_train
from src.models.predict import predict_class_image

UPLOAD_FOLDER = 'uploadFiles'
#initialize a variable with flask __name__ : referring to local python file
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#decorator -> a step before a function
@app.route("/")
def home_page():
    return render_template('home.html')

#dynamic route
@app.route("/Image_Preprocessing", methods=['GET', 'POST'])
def image_preprocessing():
    try:
        if request.method == 'POST':
            fileName = request.files.get('file')
            direcrtoryPath = request.form['text']
            madf = make_annotation_dataframe(fileName,direcrtoryPath)
            (df, annotationsTrainDF) = madf.prepare_dataframe()
            edad = eda_dataset()
            (carCountDFHigh,carCountDFLow) = edad.get_data_analysis(df)
            df = df.head(5)
            urlCarModelVar = 'static\images\image.jpg'
            urlCarLocationInImageVar = 'static\images\CheckCarLocationinImage.jpg'
            urlsampleImageWithBoudingBox = 'static\images\sampleImageWithBoudingBox.jpg'
            return render_template(
                'imagePreProcessing.html',
                tables=[df.to_html(classes='data')],
                titles=df.columns.values,
                tables_carCountDFHigh=[carCountDFHigh.to_html(classes='data')],
                title_carCountDFHigh=carCountDFHigh.columns.values,
                tables_carCountDFLow=[carCountDFLow.to_html(classes='data')],
                title_carCountDFLow=carCountDFLow.columns.values,
                rows=annotationsTrainDF.shape[0],
                columns=annotationsTrainDF.shape[1],
                filename=fileName.filename,
                direcrtoryPath=direcrtoryPath,
                url_carModel=urlCarModelVar,
                url_CarLocationInImage=urlCarLocationInImageVar,
                urlsampleImage_BoudingBox=urlsampleImageWithBoudingBox
            )
    except Exception as e:
        return render_template('error.html',
                               eMessage=e)
    return render_template('imagePreProcessing.html')


@app.route("/Image_detection", methods=['GET', 'POST'])
def image_detection():
    try:
        if request.method == 'POST':
            if 'file1' not in request.files:
                return 'there is no file1 in form!'
            file1 = request.files['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(path)
            dst_dir = "static\\images\\"
            shutil.copy(path, dst_dir)
            pci = predict_class_image()
            (act_class,pred_class)=pci.predict_and_draw('static\\images\\' + file1.filename)
            url_predicted_image_with_bounding_box = 'static\\images\\' + 'predictedSample_123.jpg'

            return render_template('ImageDetection.html',
                                   urlPredictedImage_BoudingBox=url_predicted_image_with_bounding_box,
                                   act_class=act_class,
                                   pred_class=pred_class
                                   )
    except Exception as e:
        return render_template('error.html',
                               eMessage=e)
    return render_template('ImageDetection.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route("/Model_Training", methods=['GET', 'POST'])
def model_training():
    try:
        if request.method == 'POST':
            no_of_epochs = request.form['text']

            fileName="D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\Train_Annotations.csv"
            direcrtoryPath="D:\\GreatLearning\\Capstone\\Subset\\*\\*"
            madf = make_annotation_dataframe(fileName, direcrtoryPath)
            (df, annotationsTrainDF) = madf.prepare_dataframe()
            mt = model_train()
            var_Model_train=mt.init_training(df,no_of_epochs)
            return render_template('ModelTraining.html',
                               var=var_Model_train,
                               urlTrainingGraphs='static\images\ClassAccuracyInImage.jpg'
                               )
    except Exception as e:
        return render_template('error.html',
                               eMessage=e)
    return render_template('ModelTraining.html')
