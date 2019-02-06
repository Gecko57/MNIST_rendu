from tkinter import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#  Import Data
mnist_data = pd.read_csv(os.getcwd()+'\\my_train.csv')
my_mnist_data = pd.read_csv(os.getcwd()+'\\my_train.csv')
os.getcwd()

# Full data
X = mnist_data[mnist_data.columns[1:]].values
y = mnist_data.label.values

n_train = int(2*X.shape[0]/3)

# Training data
x_train = X[:n_train, :]
y_train = y[:n_train]

# Testing Data
x_test = X[n_train:, :]
y_test = y[n_train:]

#  Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#  Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalising the data
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#  Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

#  Fitting the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 10 epochs = 10 forward pass and 10 backward pass in the NN of every batch
model.fit(x=x_train, y=y_train, epochs=10)

#  Evaluating the model and cross validation with my data
score = model.evaluate(x_test, y_test)
print('Result of Score on Keras: ' + str(int(score[1]*100)) + '%')

Matrix = np.zeros((280, 280))


class Paint(object):

    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.display = Text(self.root)
        self.text = '?'
        self.display.grid(row=0, column=1)

        self.guess_button = Button(self.root, text='Guess', command=self.guess)
        self.guess_button.grid(row=0, column=2)

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=3)

        self.choose_size_button = Scale(
            self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 10
        self.active_button = self.pen_button
        self.color = 'black'
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)
        print(self.c.find_withtag('line'))

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        global Matrix
        self.line_width = 10
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width,
                               fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
            if self.line_width <= event.x <= 280-self.line_width and self.line_width <= event.x <= 280-self.line_width:
                for i in range(int(self.line_width)):
                    for j in range(int(self.line_width)):
                        Matrix[int(event.y+(i-self.line_width/2)),
                               int(event.x+(j-self.line_width/2))] = 1

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete('all')
        global Matrix
        Matrix = np.zeros((280, 280))

    def guess(self):
        global Matrix, model
        M_bis = np.zeros((28, 28))
        for i in range(280):
            for j in range(280):
                M_bis[int(i/10), int(j/10)] += Matrix[i, j]
        for i in range(28):
            for j in range(28):
                M_bis[i, j] = min(M_bis[i, j]/25, 1)
        print(M_bis)
        y_predict = model.predict(M_bis.reshape(1, 28, 28, 1))
        y_predict = y_predict.argmax()  # Transform the vetor in a number
        print(y_predict)
        s = '{}'.format(y_predict)
        self.display.insert(END, s)
        self.display.see(END)

        plt.clf()
        image = M_bis.reshape(28, 28)
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    Paint()
