import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class eda_dataset():
    def __init__(self):
        pass

    def __getitem__(self, item):
        return getattr(self, item)

    @staticmethod
    def get_data_analysis(train_df):
        carCountDFHigh = eda_dataset.car_count_high(train_df)
        carCountDFLow = eda_dataset.car_count_low(train_df)
        eda_dataset.car_model_year_count(train_df)
        eda_dataset.image_allignment(train_df)
        eda_dataset.sample_image_display(train_df)
        return (carCountDFHigh,carCountDFLow)

    def car_count_high(train_df):
        carsCountDataFrame = eda_dataset.value_counts_df(train_df)
        return carsCountDataFrame.head(5)

    def car_count_low(train_df):
        carsCountDataFrame = eda_dataset.value_counts_df(train_df)
        return carsCountDataFrame.tail(5)

    def value_counts_df(df,self=None):
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
        df = pd.DataFrame(df['carName'].value_counts())
        df.index.name = 'carName'
        df.columns = ['count']
        return df

    def car_model_year_count(train_df):
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

    def image_allignment(train_df):
        train_df['Width_difference'] = train_df['Width'] - (train_df['xmax'] - train_df['xmin'])
        fig = plt.figure(figsize=(10, 7))
        plt.hist(train_df['Width_difference'],
                 bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                       1000])

        # Show plot
        plt.savefig('D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\static\\images\\CheckCarLocationinImage.jpg')

    def sample_image_display(train_df):
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


