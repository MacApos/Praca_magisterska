from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import adam_v2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as po
from datetime import datetime
import pandas as pd
import numpy as np
import shutil
import cv2
import os


# Wizualizacja wyników uczenia
def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines'), row=2, col=1)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title='Metrics')

    po.plot(fig, filename=os.path.join(filename, 'report.html'), auto_open=False)
    fig.write_image(os.path.join(filename, 'report.png'))

# Parametry uczenia
classes = 43
epochs = 15
learning_rate = 0.0001
batch_size = 32
input_shape = (30, 30, 3)

data_path = os.getcwd()

file1 = os.path.join(data_path, 'data_lanes.npy')
file2 = os.path.join(data_path, 'labels.npy')

if not os.path.exists(file1) or not os.path.exists(file2):
    data = []
    labels = []
    for i in range(classes):
        path = os.path.join(data_path, 'archive/Train', str(i))
        images = os.listdir(path)

        print('doing')
        for j in images:
            image = cv2.imread(path+'\\'+j)
            image = cv2.resize(image, (30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    np.save(file1, data, allow_pickle=True, fix_imports=True)
    np.save(file2, labels, allow_pickle=True, fix_imports=True)

else:
    print('not doing')

data = np.load(file1)
labels = np.load(file2)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=classes, activation='softmax'))

model.summary()

model.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('archive/Output', 'model_' + dt + '.hdf5')
model.save(filepath)

print('Eksport pliku do html')
filename = os.path.join('archive/Output', 'report' + dt + '.html')
plot_hist(history, filename=filename)

make_plot(history)
