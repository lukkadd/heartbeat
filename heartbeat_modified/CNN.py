import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SPEC_DIR = './spectograms/'
df = pd.read_csv('pre-set_b.csv')
data = []
for index, row in df.iterrows():
    data.append( (plt.imread(SPEC_DIR+ row[2].replace('.wav','.jpg')), row[3]) )

data = pd.DataFrame(data,index=None)

(x_train,y_train),(x_test,y_test) = train_test_split(data,test_size=0.3,random_state=42,stratify=data['label'])


x_train = x_train.reshape(len(x_train),400,800,3)
x_test = x_test.reshape(len(x_test),400,800,3)

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

#Training model
model.fit(x_train, y_train, verbose=0, epochs=3)

# #Testing model
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)