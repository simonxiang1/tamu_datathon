import streamlit as st
import tensorflow as tf
import numpy as np
#import matplotlib as plt

#test
tf.__version__

#loading data sets
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#for later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#preprocessing data
train_images = train_images / 255.0
test_images = test_images / 255.0

#showing first image (preprocessed)
# fig = plt.figure() 
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# plt.plot([1, 2, 3, 4, 5]) 
# st.pyplot(fig)

#plotting first 25
# fig2 = plt.figure(figsize=(10,10))
# for i in range(25):
    # plt.subplot(5,5,i+1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow(train_images[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[train_labels[i]])
# plt.show()
# st.pyplot(fig2)

#building the model through layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #flattens
    tf.keras.layers.Dense(128, activation='relu'), #good stuff
    tf.keras.layers.Dense(10) #returns logits array
])

#compiling the model
model.compile(optimizer='adam', #how the model updates
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #steering model in right direction
              metrics=['accuracy'])


#training the model
with st.spinner('Training model...'):
    model.fit(train_images, train_labels, epochs=2) #feeding model
st.success('Done!')

#calculate and display accuracy
training_loss, training_acc = model.evaluate(train_images,  train_labels, verbose=2)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
st.write('\nTraining accuracy:', training_acc)
st.write('\nTest accuracy:', test_acc)

#add softmax layer to convert logits to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#making predictions
predictions = probability_model.predict(test_images)

np.argmax(predictions[0])

probability_model.save('../src/simons_model.h5')
st.write("got here")
