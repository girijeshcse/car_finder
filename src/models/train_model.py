from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class model_train():
    def __init__(self):
        pass

    def __getitem__(self, item):
        return getattr(self, item)

    @staticmethod
    def init_training(train_df, no_of_epochs):
        import tensorflow as tf
        img_size = 224

        # Create a dictionary to hold label and corresponding class name
        num_classes = train_df['Label'].unique()

        tf.keras.backend.clear_session()
        model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,
                                                                  # Do not include FC layer at the end
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
                            metrics={'reg_op': [model_train.IoU], 'class_op': ['accuracy']})

        # Create train and test generator
        batchsize = 64
        train_generator = model_train.batch_generator(train_df, num_classes, batch_size=batchsize,
                                          img_size=img_size)  # batchsize can be changed
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

        iou = tf.py_function(model_train.calculate_iou, [y_true, y_pred], tf.float32)
        return iou