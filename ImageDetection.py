from __future__ import print_function  # In python 2.7
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pandas as pd
from flask import Flask, render_template
from flask import Flask, flash, request, redirect, url_for
import tensorflow as tf
import glob
import shutil
import os



UPLOAD_FOLDER = 'uploadFiles'
#initialize a variable with flask __name__ : referring to local python file
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#decorator -> a step before a function
@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/Image_detection", methods=['GET', 'POST'])
def image_detection():

    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        dst_dir = "static\\images\\"
        shutil.copy(path, dst_dir)
        (act_class,pred_class)=predict_and_draw('static\\images\\' + file1.filename)
        url_predicted_image_with_bounding_box = 'static\\images\\' + 'predictedSample_123.jpg'

        return render_template('ImageDetection.html',
                               urlPredictedImage_BoudingBox=url_predicted_image_with_bounding_box,
                               act_class=act_class,
                               pred_class=pred_class
                            )

    return render_template('ImageDetection.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def predict_and_draw(image):
    # fileName = request.files.get('file')
    # direcrtoryPath = request.form['text']
    direcrtoryPath = "D:\\GreatLearning\\Capstone\\TestDataSet\\*\\*"
    annotationsTrainDF = pd.read_csv("files/Train_Annotations.csv")
    annotationsTrainDF.rename(columns={'Bounding Box coordinates': 'xmin'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 2': 'ymin'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 3': 'xmax'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 4': 'ymax'}, inplace=True)
    annotationsTrainDF.rename(columns={'Image class': 'Image_class'}, inplace=True)
    annotationsTrainDF.rename(columns={'Image Name': 'Image_Name'}, inplace=True)
    dfTrain = readImagesFromDirectory(direcrtoryPath)
    train_df = dfTrain.merge(annotationsTrainDF, how='inner', left_on='imageName', right_on='Image_Name')
    train_df = train_df.assign(image_path=direcrtoryPath)
    train_df['image_path_trimmed'] = train_df['image_path'].astype(str).str[:-4]
    train_df['File'] = train_df['image_path_trimmed'] + "\\" + train_df['carName'] + "\\" + train_df['imageName']
    train_df = EDApart4_LabelEncoding(train_df)
    train_df.rename(columns={'Image_class': 'Class'}, inplace=True)
    train_df.rename(columns={'le_carName': 'Label'}, inplace=True)

    # Create a dictionary to hold label and corresponding class name
    num_classes_test = train_df['Label'].unique()
    label_class_dict = dict(zip(train_df['Label'], train_df['carName']))

    filenameVar = (os.path.basename(image.replace('\\', os.sep)))
    image_num = train_df.loc[train_df['imageName'] == filenameVar].index.tolist()[0]

    img_size = 224
    # Load image
    # image_num=98
    img = tf.keras.preprocessing.image.load_img(train_df.loc[image_num, 'File'])
    w, h = img.size

    # Prepare input for model
    # 1. Resize image
    img_resized = img.resize((img_size, img_size))
    # 2. Conver to array and make it a batch of 1
    input_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    input_array = np.expand_dims(input_array, axis=0)
    # 3. Normalize image data
    input_array = tf.keras.applications.efficientnet.preprocess_input(input_array)
    final_model = tf.keras.models.load_model("model\Cars_196_dataset_localization_Adam_EfficientNet_E25_2022_01_14.h5", compile=False,custom_objects={"iou": IoU})

    # Prediction
    pred = final_model.predict(input_array)
    # Get classification and regression predictions
    label_pred, bbox_pred = pred[0][0], pred[1][0]
    # Get Label with highest probability
    pred_class = label_class_dict[np.argmax(label_pred)]

    # Read actual label and bounding box
    act_class = train_df.loc[image_num, 'Class']
    act_class = train_df.loc[image_num, 'carName']
    xmin, ymin, xmax, ymax = train_df.loc[image_num, ['xmin', 'ymin', 'xmax', 'ymax']]

    # Draw bounding boxes - Actual (Red) and Predicted(Green)
    img = cv2.imread(train_df.loc[image_num, 'File'])

    # Draw actual bounding box - Red
    img = cv2.rectangle(img, (xmin, ymin),
                        (xmax, ymax), (0, 0, 255), 3)

    # Draw predicted bounding box -  Green
    img = cv2.rectangle(img, (int(bbox_pred[0] * w), int(bbox_pred[1] * h)),
                        (int((bbox_pred[0] + bbox_pred[2]) * w), int((bbox_pred[1] + bbox_pred[3]) * h)), (0, 255, 0), 3
                        )

    # Display the picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imsave('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\predictedSample_123.jpg', img)
    return(act_class,pred_class)

@app.route("/Model_Training", methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        no_of_epochs = request.form['text']
        import pandas as pd
        annotationsTrainDF = pd.read_csv("D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\Train_Annotations.csv")
        annotationsTrainDF.rename(columns={'Bounding Box coordinates': 'xmin'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 2': 'ymin'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 3': 'xmax'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 4': 'ymax'}, inplace=True)
        annotationsTrainDF.rename(columns={'Image class': 'Image_class'}, inplace=True)
        annotationsTrainDF.rename(columns={'Image Name': 'Image_Name'}, inplace=True)

        # /content/drive/MyDrive/MachineLearning/CapstoneProject/Dataset_1/Dataset/Car Images/Samples/Jeep Liberty SUV 2012/00271.jpg
        trainDataSetPath = "D:\\GreatLearning\\Capstone\\Subset\\*\\*"
        # reading png files in the path
        trainList = glob.glob(trainDataSetPath)

        trainListPaths = []
        for fileandFolder in trainList:
            lList = fileandFolder.split("\\")[-2:]
            trainListPaths.append(lList)

        dfTrain = pd.DataFrame(trainListPaths, columns=['carName', 'imageName'])
        dfTrain['carModel'] = dfTrain['carName'].str[-4:]
        dfTrain['carModel_1'] = dfTrain['carName'].str[:-4]

        train_df = dfTrain.merge(annotationsTrainDF, how='inner', left_on='imageName', right_on='Image_Name')
        train_df = train_df.assign(image_path='D:\\GreatLearning\\Capstone\\Subset\\')

        from sklearn import preprocessing
        # label_encoder object knows how to understand word labels.
        label_encoder = preprocessing.LabelEncoder()
        train_df['le_carName'] = label_encoder.fit_transform(train_df['carName'])

        dfTrain_W_H = pd.read_csv("D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\train_8144_images.csv")
        dfTrain_W_H = dfTrain_W_H[['Height', 'Width', 'Image_Name']]
        train_df = train_df.merge(dfTrain_W_H, how='inner', left_on='imageName', right_on='Image_Name')
        train_df['File'] = train_df['image_path'] + "/" + train_df['carName'] + "/" + train_df['imageName']
        train_df.rename(columns={'Image_class': 'Class'}, inplace=True)
        train_df.rename(columns={'le_carName': 'Label'}, inplace=True)

        var_Model_train=init_training(train_df,no_of_epochs)
        return render_template('ModelTraining.html',
                           var=var_Model_train,
                           urlTrainingGraphs='static\images\ClassAccuracyInImage.jpg'
                            )
    return render_template('ModelTraining.html')

def init_training(train_df,no_of_epochs):
    import tensorflow as tf
    img_size = 224

    # Create a dictionary to hold label and corresponding class name
    num_classes = train_df['Label'].unique()

    tf.keras.backend.clear_session()
    model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,  # Do not include FC layer at the end
                                                              input_shape=(img_size, img_size, 3),
                                                              weights='imagenet')
    for layer in model.layers:
        layer.trainable = False

    # get Output layer of Pre-trained model
    x1 = model.output

    # Flatten the output to feed to Dense layer
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)

    # Add one Dense layer
    x3 = tf.keras.layers.Dense(2560, activation='relu')(x2)

    # Batch Norm
    x5 = tf.keras.layers.BatchNormalization()(x3)

    # Classification
    label_output = tf.keras.layers.Dense(len(num_classes),
                                         activation='softmax',
                                         name='class_op')(x5)
    # Regression
    bbox_output = tf.keras.layers.Dense(4,
                                        activation='sigmoid',
                                        name='reg_op')(x5)

    # Non Sequential model as it has two different outputs
    final_model = tf.keras.models.Model(inputs=model.input,  # Pre-trained model input as input layer
                                        outputs=[label_output, bbox_output])  # Output layer added

    optimizerVar = tf.keras.optimizers.Adam()
    final_model.compile(optimizer=optimizerVar,
                        loss={'reg_op': 'mse', 'class_op': 'categorical_crossentropy'},
                        metrics={'reg_op': [IoU], 'class_op': ['accuracy']})

    # Create train and test generator
    batchsize = 64
    train_generator = batch_generator(train_df,num_classes, batch_size=batchsize,img_size=img_size)  # batchsize can be changed
    # test_generator = batch_generator(val_df, batch_size=batchsize)

    checkpoint_filepath = 'D:\\Temp\\CapstoneProject\\tempCSV\\checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='class_op_accuracy',
        mode='max',
        save_best_only=True)

    history_1 = final_model.fit(train_generator,
                                epochs=int(no_of_epochs),
                                steps_per_epoch=train_df.shape[0] // batchsize,
                                # validation_data=test_generator,
                                # validation_steps = val_df.shape[0]//batchsize,
                                callbacks=[model_checkpoint_callback]
                                )

    acc = history_1.history['class_op_accuracy']
    iou = history_1.history['reg_op_IoU']
    epochs_range = range(int(no_of_epochs))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, iou, label='Training IOU')
    plt.legend(loc='upper right')
    plt.title('Training and Validation IOU')
    plt.savefig('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\ClassAccuracyInImage.jpg')

    return history_1.history['class_op_accuracy']


def batch_generator(df, num_classes, batch_size=32,img_size=224):

    while True:

        # Create indexes
        image_nums = np.random.randint(0, df.shape[0], size=batch_size)

        # Create empty arrays
        # 1. To hold image input
        batch_images = np.zeros(shape=(batch_size, img_size, img_size, 3))

        # Classification Labels
        batch_labels = np.zeros(shape=(batch_size, len(num_classes)))

        # Regression labels - 4 numbers per example image
        batch_bboxes = np.zeros(shape=(batch_size, 4))

        for i in range(batch_size):
            # Read image and resize
            img = tf.keras.preprocessing.image.load_img(df.loc[image_nums[i], 'File'],
                                                        target_size=(img_size, img_size))

            # Conver to numpy array
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            # Update batch
            batch_images[i] = img_array

            # Read image classification label & convert to one hot vector
            cl_label = df.loc[image_nums[i], 'Label']
            cl_label = tf.keras.utils.to_categorical(cl_label, num_classes=len(num_classes))
            batch_labels[i] = cl_label

            # Read and resize bounding box co-ordinates
            img_width = df.loc[image_nums[i], 'Width']
            img_height = df.loc[image_nums[i], 'Height']

            xmin = df.loc[image_nums[i], 'xmin'] * img_size / img_width
            xmax = df.loc[image_nums[i], 'xmax'] * img_size / img_width

            ymin = df.loc[image_nums[i], 'ymin'] * img_size / img_height
            ymax = df.loc[image_nums[i], 'ymax'] * img_size / img_height

            # We will ask model to predict xmin, ymin, width and height of bounding box
            batch_bboxes[i] = [xmin, ymin, xmax - xmin, ymax - ymin]

        # Normalize batch images as per Pre-trained model to be used
        for i in range(batch_size):
            batch_images[i] = tf.keras.applications.efficientnet.preprocess_input(batch_images[i])

        # Make bounding boxes (x, y, w, h) as numbers between 0 and 1 - this seems to work better
        batch_bboxes = batch_bboxes / img_size

        # Return batch - use yield function to make it a python generator
        yield batch_images, [batch_labels, batch_bboxes]


#dynamic route
@app.route("/Image_Preprocessing", methods=['GET', 'POST'])
def image_preprocessing():
    df = pd.DataFrame()

    if request.method == 'POST':
        fileName = request.files.get('file')
        direcrtoryPath = request.form['text']
        (df, annotationsTrainDF, carCountDFHigh, carCountDFLow) = readCSVFileAsDataFrame(fileName,direcrtoryPath)
        urlCarModelVar='static\images\image.jpg'
        urlCarLocationInImageVar='static\images\CheckCarLocationinImage.jpg'
        urlsampleImageWithBoudingBox='static\images\sampleImageWithBoudingBox.jpg'
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
    return render_template('imagePreProcessing.html')

def readCSVFileAsDataFrame(fileName,direcrtoryPath):
    annotationsTrainDF = pd.read_csv(fileName)
    annotationsTrainDF.rename(columns={'Bounding Box coordinates': 'xmin'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 2': 'ymin'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 3': 'xmax'}, inplace=True)
    annotationsTrainDF.rename(columns={'Unnamed: 4': 'ymax'}, inplace=True)
    annotationsTrainDF.rename(columns={'Image class': 'Image_class'}, inplace=True)
    annotationsTrainDF.rename(columns={'Image Name': 'Image_Name'}, inplace=True)
    dfTrain = readImagesFromDirectory(direcrtoryPath)
    train_df = dfTrain.merge(annotationsTrainDF, how='inner', left_on='imageName', right_on='Image_Name')
    train_df = train_df.assign(image_path=direcrtoryPath)
    train_df['image_path_trimmed'] = train_df['image_path'].astype(str).str[:-4]
    train_df['File'] = train_df['image_path_trimmed'] + "\\" + train_df['carName'] + "\\" + train_df['imageName']
    carCountDFHigh = EDApart1(train_df)
    carCountDFLow = EDApart2(train_df)
    EDApart3(train_df)
    EDApart4_1(train_df)
    EDApart5(train_df)
    return (train_df.head(5), annotationsTrainDF, carCountDFHigh, carCountDFLow)

def readImagesFromDirectory(path):
    trainDataSetPath = path#"/content/drive/MyDrive/MachineLearning/CapstoneProject/Dataset_1/Dataset/Car Images/Split/train/*/*"
    # reading png files in the path
    trainList = glob.glob(trainDataSetPath)

    trainListPaths = []
    for fileandFolder in trainList:
        lList = fileandFolder.split("\\")[-2:]
        trainListPaths.append(lList)

    dfTrain = pd.DataFrame(trainListPaths, columns=['carName', 'imageName'])
    dfTrain['carModel'] = dfTrain['carName'].str[-4:]
    dfTrain['carModel_1'] = dfTrain['carName'].str[:-4]

    return dfTrain

def value_counts_df(df, col):
    """
    Returns pd.value_counts() as a DataFrame

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe on which to run value_counts(), must have column `col`.
    col : str
        Name of column in `df` for which to generate counts

    Returns
    -------
    Pandas Dataframe
        Returned dataframe will have a single column named "count" which contains the count_values()
        for each unique value of df[col]. The index name of this dataframe is `col`.

    """
    df = pd.DataFrame(df[col].value_counts())
    df.index.name = col
    df.columns = ['count']
    return df

def EDApart1(train_df):
    carsCountDataFrame = value_counts_df(train_df, 'carName')
    return carsCountDataFrame.head(5)

def EDApart2(train_df):
    carsCountDataFrame = value_counts_df(train_df, 'carName')
    return carsCountDataFrame.tail(5)

def EDApart3(train_df):
    # train_df['carName'].value_counts().plot(kind='bar')

    carModel = train_df['carModel'].value_counts().index
    carCounts = train_df['carModel'].value_counts()

    # Figure Size
    fig = plt.figure(figsize=(10, 7))

    plt.title('Car count in dataset vs model', fontdict={'fontsize': 20})
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Count', fontsize=16)

    # Horizontal Bar Plot
    plt.bar(carModel, carCounts)

    plt.plot([1, 2])
    plt.savefig('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\carModelYearCount.jpg')


def EDApart4_LabelEncoding(train_df):
    from sklearn import preprocessing
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    train_df['le_carName'] = label_encoder.fit_transform(train_df['carName'])
    return train_df

def EDApart4_1(train_df):
    train_df = EDApart4_LabelEncoding(train_df)
    dfTrain_W_H = pd.read_csv("D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\train_8144_images.csv")
    dfTrain_W_H = dfTrain_W_H[['Height', 'Width', 'Image_Name']]
    train_df = train_df.merge(dfTrain_W_H, how='inner', left_on='imageName', right_on='Image_Name')
    train_df['File'] = train_df['image_path'] + "/" + train_df['carName'] + "/" + train_df['imageName']
    train_df.rename(columns={'Image_class': 'Class'}, inplace=True)
    train_df.rename(columns={'le_carName': 'Label'}, inplace=True)

    train_df['Width_difference'] = train_df['Width'] - (train_df['xmax'] - train_df['xmin'])
    fig = plt.figure(figsize=(10, 7))
    plt.hist(train_df['Width_difference'], bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])

    # Show plot
    plt.savefig('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\CheckCarLocationinImage.jpg')

def EDApart5(train_df):
    dfTrain_W_H = pd.read_csv("D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\train_8144_images.csv")
    dfTrain_W_H = dfTrain_W_H[['Height', 'Width', 'Image_Name']]
    train_df = train_df.merge(dfTrain_W_H, how='inner', left_on='imageName', right_on='Image_Name')

    # Pickup a random image number
    img_num = np.random.randint(0, train_df.shape[0])
    # Read the image and draw a rectangle as per bounding box information
    img = cv2.imread(train_df.loc[img_num, 'File'])
    img = cv2.resize(img, (224, 224))
    w = train_df.loc[img_num, 'Width']
    h = train_df.loc[img_num, 'Height']
    x_ratio = 224 / w
    y_ratio = 224 / h
    cv2.rectangle(img,
                  (int(train_df.loc[img_num, 'xmin'] * x_ratio), int(train_df.loc[img_num, 'ymin'] * y_ratio)),
                  (int(train_df.loc[img_num, 'xmax'] * x_ratio), int(train_df.loc[img_num, 'ymax'] * y_ratio)),
                  (0, 255, 0),
                  2)

    # Convert BGR format (used by opencv to RGB format used by matplotlib)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw image using matplotlib
    plt.suptitle(train_df.loc[img_num, 'carModel_1'])
    # plt.savefig('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\sampleImageWithBoudingBox.jpg')
    plt.imsave('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\sampleImageWithBoudingBox.jpg', img)

def calculate_iou(y_true, y_pred):
        """
        Input:
        Keras provides the input as numpy arrays with shape (batch_size, num_columns).

        Arguments:
        y_true -- first box, numpy array with format [x, y, width, height, conf_score]
        y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
        x any y are the coordinates of the top left corner of each box.

        Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)

        """

        results = []

        for i in range(0, y_true.shape[0]):
            # set the types so we are sure what type we are using
            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)

            # boxTrue
            x_boxTrue_tleft = y_true[i, 0]  # numpy index selection
            y_boxTrue_tleft = y_true[i, 1]
            boxTrue_width = y_true[i, 2]
            boxTrue_height = y_true[i, 3]
            area_boxTrue = (boxTrue_width * boxTrue_height)

            # boxPred
            x_boxPred_tleft = y_pred[i, 0]
            y_boxPred_tleft = y_pred[i, 1]
            boxPred_width = y_pred[i, 2]
            boxPred_height = y_pred[i, 3]
            area_boxPred = (boxPred_width * boxPred_height)

            # calculate the bottom right coordinates for boxTrue and boxPred

            # boxTrue
            x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
            y_boxTrue_br = y_boxTrue_tleft + boxTrue_height  # Version 2 revision

            # boxPred
            x_boxPred_br = x_boxPred_tleft + boxPred_width
            y_boxPred_br = y_boxPred_tleft + boxPred_height  # Version 2 revision

            # calculate the top left and bottom right coordinates for the intersection box, boxInt

            # boxInt - top left coords
            x_boxInt_tleft = np.max([x_boxTrue_tleft, x_boxPred_tleft])
            y_boxInt_tleft = np.max([y_boxTrue_tleft, y_boxPred_tleft])  # Version 2 revision

            # boxInt - bottom right coords
            x_boxInt_br = np.min([x_boxTrue_br, x_boxPred_br])
            y_boxInt_br = np.min([y_boxTrue_br, y_boxPred_br])

            # Calculate the area of boxInt, i.e. the area of the intersection
            # between boxTrue and boxPred.
            # The np.max() function forces the intersection area to 0 if the boxes don't overlap.

            # Version 2 revision
            area_of_intersection = \
                np.max([0, (x_boxInt_br - x_boxInt_tleft)]) * np.max([0, (y_boxInt_br - y_boxInt_tleft)])

            iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)

            # This must match the type used in py_func
            iou = np.array(iou, dtype=np.float32)

            # append the result to a list at the end of each loop
            results.append(iou)

        # return the mean IoU score for the batch
        return np.mean(results)

def IoU(y_true, y_pred):
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours
    # trying to debug and almost give up.

    iou = tf.py_function(calculate_iou, [y_true, y_pred], tf.float32)
    return iou

        # df.to_html(header="true", table_id="table")
            #return render_template('imagePreProcessing.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    #return render_template('imagePreProcessing.html')
            #return render_template('imagePreProcessing.html', car_name="Lamborghini Diablo Coupe 2001")


