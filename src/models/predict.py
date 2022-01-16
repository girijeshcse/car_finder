from __future__ import print_function
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from src.features.build_features import make_annotation_dataframe

class predict_class_image():
    def __init__(self):
        pass

    def __getitem__(self, item):
        return getattr(self, item)

    def batch_generator(df, num_classes, batch_size=32, img_size=224):

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

        iou = tf.py_function(predict_class_image.calculate_iou, [y_true, y_pred], tf.float32)
        return iou

    @staticmethod
    def predict_and_draw(image):
        fileName = "D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\Train_Annotations.csv"
        direcrtoryPath = "D:\\GreatLearning\\Capstone\\Subset\\*\\*"
        madf = make_annotation_dataframe(fileName, direcrtoryPath)
        (train_df, annotationsTrainDF) = madf.prepare_dataframe()
        num_classes = train_df['Label'].unique()
        label_class_dict = dict(zip(train_df['Label'], train_df['carName']))


        filenameVar = (os.path.basename(image.replace('\\', os.sep)))
        image_num = train_df.loc[train_df['imageName'] == filenameVar].index.tolist()[0]

        img_size = 224
        # Load image
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

        final_model = tf.keras.models.load_model(
            "model\Cars_196_dataset_localization_Adam_EfficientNet_E25_2022_01_14.h5", compile=False,
            custom_objects={"iou": predict_class_image.IoU})

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
                            (int((bbox_pred[0] + bbox_pred[2]) * w), int((bbox_pred[1] + bbox_pred[3]) * h)),
                            (0, 255, 0), 3
                            )
        # Display the picture
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\predictedSample_123.jpg', img)
        return (act_class, pred_class)

