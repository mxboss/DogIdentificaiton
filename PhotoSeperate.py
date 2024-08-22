import tensorflow as tf
import os
import  tensorflow_hub as hub
import torch
import numpy as np
from tensorflow import keras
from keras import layers
from spyder.utils.qthelpers import get_image_label
from xlwings.utils import process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model_path='E:/workSpace/PythonStudy/PhotoSeperate/20240717-180059_full-image-set-mobilev2-Adam.h5'

def load_model(model_path):
    """
    Loads a saved model from a specified path
    """
    print(f"Loading saved model from: {model_path}")

    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model

load_full_model = load_model(
    "E:/workSpace/PythonStudy/PhotoSeperate/20240717-180059_full-image-set-mobilev2-Adam.h5"
)


# custom_path = "./photo"
# custom_image_path = [custom_path + fname for fname in os.listdir(custom_path)]
# print(custom_image_path)
#
#
# # define batch size
# BATCH_SIZE = 32
#
# # Create a function to turn data into batches
# def create_data_batches(
#     X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
# ):
#     """
#     Creates batches of data out of image (X) and label (y) pairs.
#     Shuffles the data if its training data but dosent shuffle if its validatoin data.
#     Also accepts test data as input (no labels).
#     """
#     # If data is test data set, we probably dont have labels
#     if test_data:
#         print("Creating test data batches...")
#         data = tf.data.Dataset.from_tensor_slices(
#             (tf.constant(X))
#         )  # only file paths and no labels
#         data_batch = data.map(process_image).batch(BATCH_SIZE)
#         return data_batch
#
#     # If data is a valid dataset, we dont need to shuffle it
#     elif valid_data:
#         print("Creating validation data batches...")
#         data = tf.data.Dataset.from_tensor_slices(
#             (tf.constant(X), tf.constant(y))  # filepaths
#         )  # labels
#         data_batch = data.map(get_image_label).batch(BATCH_SIZE)
#         return data_batch
#     else:
#         print("Creating training data batches")
#         # Turn filepaths and labels into Tensors
#         data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
#
#         # Shuffling labels and pathnames before mapping, image processor function is faster than shuffling images
#         data = data.shuffle(buffer_size=len(X))
#
#         # Create (image, label) tuples (this also turns the image path into preprocessed image)
#         data = data.map(get_image_label)
#
#         # Turn the training data into batches
#         data_batch = data.batch(BATCH_SIZE)
#
#     return data_batch
#
# # Create a function to return a tuple (image, label)
# def get_image_label(image_path, label):
#     image = process_image(image_path)
#     return image, label
#
# # Defining image size
# IMG_SIZE = 224
#
#
# # Creating function to preprocess images
# def process_image(image_path, img_size=IMG_SIZE):
#     """
#     Takes an image file path and turns the image into a Tensor
#     """
#     # Read image file
#     image = tf.io.read_file(image_path)
#
#     # Turn image into numerical tensor with 3 color channel(RGB)
#     image = tf.image.decode_jpeg(image, channels=3)
#
#     # Convert color channel value from 0-255 to 0-1 values
#     image = tf.image.convert_image_dtype(image, tf.float32)
#
#     # Resize the image
#     image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
#
#     return image
#
# custom_data = create_data_batches(custom_image_path, test_data=True)
# # Make predictions on custom data
# custom_preds = load_full_model.predict(custom_data)
#
# # Turn prediction probabilities into their respective label (easier to understand)
# def get_pred_label(prediction_probabilities):
#     """
#     Turns an array of prediction probabilities into a label.
#     """
#     return np.argmax(prediction_probabilities)
#
# # Get custom image prediction labels
# custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
# custom_pred_labels





