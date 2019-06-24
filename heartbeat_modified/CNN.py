import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
SPEC_DIR = './spectograms/'
df = pd.read_csv('pre-set_b.csv')
X = []
Y = []
for index, row in df.iterrows():
    X.append(plt.imread(SPEC_DIR+ row[2].replace('.wav','.jpg')))
    Y.append(row[3])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

#reshaping images
length = len(x_train)
x_train = np.array(x_train)
x_train = x_train.reshape(length,400,800,3)

length = len(x_test)
x_test = np.array(x_test)
x_test = x_test.reshape(length,400,800,3)

#categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

#Building model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(400,800,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#Compiling model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('******************************************************************************')
#Training model
model.fit(x_train, y_train, verbose=0, epochs=3)

model.summary()


print('******************************************************************************')
# #Testing model
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print(y_train_pred)
print(y_test_pred)