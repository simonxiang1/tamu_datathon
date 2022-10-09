import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#test
tf.__version__


#loading data sets
#fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#for later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#generate data sets
image_size = (128, 128)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#preprocessing data
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# train_ds = train_ds / 255.0
# val_ds = val_ds / 255.0
resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(128, 128),
  tf.keras.layers.Rescaling(1./255)
])


#showing first image (preprocessed)
# fig = plt.figure() 
# plt.imshow(train_ds[0])
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
    # plt.imshow(train_ds[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[train_labels[i]])
# plt.show()
# st.pyplot(fig2)

fig = plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()
st.pyplot(fig)

#building the model through layers
model = tf.keras.Sequential([
    resize_and_rescale,
    tf.keras.layers.Flatten(input_shape=(128, 128)), #flattens
    tf.keras.layers.Dense(128, activation='relu'), #good stuff
    tf.keras.layers.Dense(10) #returns logits array
])

#compiling the model
# model.compile(optimizer='adam', #how the model updates
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #steering model in right direction
              # metrics=['accuracy'])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

#training the model
with st.spinner('Training model...'):
    model.fit(train_ds, epochs=2, validation_data = val_ds) #feeding model
st.success('Done!')

#calculate and display accuracy
training_loss, training_acc = model.evaluate(train_ds, verbose=2)
val_loss, val_acc = model.evaluate(val_ds, verbose=2)
st.write('\nTraining accuracy:', training_acc)
st.write('\nTest accuracy:', val_acc)

#add softmax layer to convert logits to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#making predictions
predictions = probability_model.predict(val_ds)

np.argmax(predictions[0])

probability_model.save('../src/simons_model.h5')
st.write("got here")
