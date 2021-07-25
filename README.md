# CNN_traffic_sign_classification
Convolutional Neural Network(CNN) model that can classify the given traffic sign image to one of the 43 different traffic sign classes.

Theory:
    There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.The main aim of the project is to build a deep neural network model that classifies the traffic signs present in the image into different categories. With this model, autonomous vehicles can identify and understand the traffic signs.For this project kaggle is used for gathering traffic signs dataset, a input image with traffic sign in it is fed to the Deep - learning model which will classify what traffic sign it is.

MODEL ARCHITECTURE
     We are building the model with Keras using Convolutional Neural Networks(CNN) activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.
     
The CNN model architecture consists of the following layers:

1)Convolution 2D Layer : filter= 32 ,kernel size = (5,5)

2)Convolution 2D Layer : filter= 32 ,kernel size = (5,5)

3)Maxpool 2D : pool size = (2,2)

4)Dropout  : rate = 0.25

5)Convolution 2D Layer : filter= 64 ,kernel size = (3,3)

6)Convolution 2D Layer : filter= 64 ,kernel size = (3,3)

7)Maxpool 2D : pool size = (2,2)

8)Dropout  : rate = 0.25

9)Flatten :  Flattens the layers into 1Dimension

10)Dense fully connected layer : nodes = 256

11)Dropout layer : rate = 0.5

Training: 
          Download the german traffic sign dataset from kaggle website(https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).In the dataset, ‘train’ folder contains 43 folders each representing a different class. The range of the folder is from 0 to 42. With the help of the OS module iterate over all the classes and append images and their respective labels in the data and labels list.
          The PIL library is used to open image content into an array. Finally  all the images and their corresponding labels stored in lists (data and labels).Convert the list into numpy arrays to feed  to the model. 
          The shape of data is (27310, 30, 30, 3) which means that there are 27310 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value). With the sklearn package,use the train_test_split() method to split training and testing data.
      Build the model and fit the above training and testing  data to the model and check the training loss and testing loss.
    
Testing:
        The Kaggle dataset consists of "test" folder where 10000+ images are available for testing.Resize the images to 30 * 30  to feed them as input to the deep learning model.Store the predicted labels of the images predicted by the model and true labels in two lists.
        Now use accuracy_score function to know the accuracy of the model in predicting the images.
        Use confusion matrix to check  each class accuracy.

Conclusions:
            The model acheived a 95% accuracy in classifying the traffic signs.
            
