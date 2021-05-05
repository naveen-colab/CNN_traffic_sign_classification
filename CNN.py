import numpy as np 
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

#Get the german traffic sign images dataset from kaggle 
data = []                
labels = []
classes = 43      #dataset consists of 43 different classes       
#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))  #curr_path is the path to the data set folder,'train' is the folder name.
    images = os.listdir(path)
    k=len(images)
    print(len(images))
    for a in images:
              image = Image.open(path + '//'+ a)
              image = image.resize((30,30))      #resize all the images to 30*30 for uniformity in all the dataset images
              image = np.array(image)
              data.append(image)
              labels.append(i)
#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("matrix_model.h5")

model.evaluate(X_test,y_test)                           #evaluates the model accuracy



#confusion matrix 
def plot_matrix(y_true,pred_y):
    matrix_labels = unique_labels(k)
    matrix_columns = [f'Predicted{label}' for label in matrix_labels]
    matrix_indexs = [f'Actual {label}' for label in matrix_labels]
    matrix_table = pd.DataFrame(confusion_matrix(y_true,pred_y),column = matrix_columns,row =matrix_indexs)
    return matrix_table
  
confusion_matrix1 = plot_matrix(k,test_labels)
print(classification_repor(y_true,pred_y))
