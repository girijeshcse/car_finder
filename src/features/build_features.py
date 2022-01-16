import pandas as pd
import glob

class make_annotation_dataframe():
    def __init__(self, fileName, direcrtoryPath):
        self.fileName = fileName
        self.direcrtoryPath = direcrtoryPath

    def __getitem__(self, item):
        return getattr(self, item)

    def prepare_dataframe(self):
        annotationsTrainDF = self.readCSVFileAsDataFrame()
        dfTrain = self.readImagesFromDirectory()
        train_df = dfTrain.merge(annotationsTrainDF, how='inner', left_on='imageName', right_on='Image_Name')
        train_df = train_df.assign(image_path=self.direcrtoryPath)

        from sklearn import preprocessing
        # label_encoder object knows how to understand word labels.
        label_encoder = preprocessing.LabelEncoder()
        train_df['le_carName'] = label_encoder.fit_transform(train_df['carName'])

        dfTrain_W_H = pd.read_csv("D:\\GreatLearning\\Flask\\OBJECT_DETECTION_CAR\\files\\train_8144_images.csv")
        dfTrain_W_H = dfTrain_W_H[['Height', 'Width', 'Image_Name']]
        train_df = train_df.merge(dfTrain_W_H, how='inner', left_on='imageName', right_on='Image_Name')

        train_df.rename(columns={'Image_class': 'Class'}, inplace=True)
        train_df.rename(columns={'le_carName': 'Label'}, inplace=True)

        train_df['image_path_trimmed'] = train_df['image_path'].astype(str).str[:-4]
        train_df['File'] = train_df['image_path_trimmed'] + "\\" + train_df['carName'] + "\\" + train_df['imageName']

        return (train_df, annotationsTrainDF)


    def readCSVFileAsDataFrame(self=None):
        annotationsTrainDF = pd.read_csv(self.fileName)
        annotationsTrainDF.rename(columns={'Bounding Box coordinates': 'xmin'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 2': 'ymin'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 3': 'xmax'}, inplace=True)
        annotationsTrainDF.rename(columns={'Unnamed: 4': 'ymax'}, inplace=True)
        annotationsTrainDF.rename(columns={'Image class': 'Image_class'}, inplace=True)
        annotationsTrainDF.rename(columns={'Image Name': 'Image_Name'}, inplace=True)
        return annotationsTrainDF


    def readImagesFromDirectory(self=None):
        trainDataSetPath = self.direcrtoryPath  # "/content/drive/MyDrive/MachineLearning/CapstoneProject/Dataset_1/Dataset/Car Images/Split/train/*/*"
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
